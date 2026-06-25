#can be used individually to get benign vs malicious classification, but not for multi-class classification
import os
import logging
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import normalized_mutual_info_score, accuracy_score
from sklearn.cluster import KMeans
from collections import Counter
from imblearn.over_sampling import SMOTE


def safe_save(model, path):
    try:
        model.save(path)
    except Exception:
        pass


def safe_dump(obj, path):
    try:
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
    except Exception:
        pass


data_dir   = os.getenv('DATA_DIR',    '/platform/swarmml/data')
model_dir  = os.getenv('MODEL_DIR',   '/platform/swarmml/model')
max_epochs = int(os.getenv('MAX_EPOCHS', '100'))
min_peers  = int(os.getenv('MIN_PEERS',  '2'))

col_names = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root',
    'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty',
]

attack_map = {
    'normal':         'normal',
    'back':           'dos',
    'land':           'dos',
    'neptune':        'dos',
    'pod':            'dos',
    'smurf':          'dos',
    'teardrop':       'dos',
    'mailbomb':       'dos',
    'apache2':        'dos',
    'processtable':   'dos',
    'udpstorm':       'dos',
    'ipsweep':        'probe',
    'nmap':           'probe',
    'portsweep':      'probe',
    'satan':          'probe',
    'mscan':          'probe',
    'saint':          'probe',
    'ftp_write':      'r2l',
    'guess_passwd':   'r2l',
    'imap':           'r2l',
    'multihop':       'r2l',
    'phf':            'r2l',
    'spy':            'r2l',
    'warezclient':    'r2l',
    'warezmaster':    'r2l',
    'sendmail':       'r2l',
    'named':          'r2l',
    'snmpgetattack':  'r2l',
    'snmpguess':      'r2l',
    'xlock':          'r2l',
    'xsnoop':         'r2l',
    'httptunnel':     'r2l',
    'buffer_overflow':'u2r',
    'loadmodule':     'u2r',
    'perl':           'u2r',
    'rootkit':        'u2r',
    'ps':             'u2r',
    'sqlattack':      'u2r',
    'xterm':          'u2r',
}

PROTOCOL_CATS = ['icmp', 'tcp', 'udp']

SERVICE_CATS = [
    'IRC', 'X11', 'Z39_50', 'aol', 'auth', 'bgp', 'courier', 'csnet_ns',
    'ctf', 'daytime', 'discard', 'domain', 'domain_u', 'echo', 'eco_i',
    'ecr_i', 'efs', 'exec', 'finger', 'ftp', 'ftp_data', 'gopher', 'harvest',
    'hostnames', 'http', 'http_2784', 'http_443', 'http_8001', 'imap4',
    'iso_tsap', 'klogin', 'kshell', 'ldap', 'link', 'login', 'mtp', 'name',
    'netbios_dgm', 'netbios_ns', 'netbios_ssn', 'netstat', 'nnsp', 'nntp',
    'ntp_u', 'other', 'pm_dump', 'pop_2', 'pop_3', 'printer', 'private',
    'red_i', 'remote_job', 'rje', 'shell', 'smtp', 'sql_net', 'ssh', 'sunrpc',
    'supdup', 'systat', 'telnet', 'tftp_u', 'tim_i', 'time', 'urh_i', 'urp_i',
    'uucp', 'uucp_path', 'vmnet', 'whois',
]

FLAG_CATS = ['OTH', 'REJ', 'RSTO', 'RSTOS0', 'RSTR', 'S0', 'S1', 'S2', 'S3', 'SF', 'SH']


print("Loading dataset...")
fp = os.path.join(data_dir, 'KDDTrain+.txt')

with open(fp) as f:
    first = f.readline().strip()

if first.startswith('duration'):
    df    = pd.read_csv(fp, low_memory=False)
    y_str = df['label'].values
    df    = df.drop(['label'], axis=1)
else:
    df          = pd.read_csv(fp, names=col_names, low_memory=False)
    df['label'] = df['label'].map(attack_map).fillna('unknown')
    y_str       = df['label'].values
    df          = df.drop(['label', 'difficulty'], axis=1)

cat_cols = ['protocol_type', 'service', 'flag']
num_cols = [c for c in df.columns if c not in cat_cols]

prep = ColumnTransformer([
    ('cat', OneHotEncoder(
        categories=[PROTOCOL_CATS, SERVICE_CATS, FLAG_CATS],
        sparse_output=False,
        handle_unknown='ignore'
    ), cat_cols),
    ('num', StandardScaler(), num_cols),
])

X = prep.fit_transform(df).astype('float32')

os.makedirs(model_dir, exist_ok=True)
safe_dump(prep, os.path.join(model_dir, 'preprocessor.pkl'))


def target_dist(q):
    w = q ** 2 / q.sum(axis=0)
    return (w.T / w.sum(axis=1)).T


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

in_dim  = X.shape[1]
enc_dim = 64
BS      = 256
K       = 25

ei  = keras.Input(shape=(in_dim,))
e   = layers.Dense(256)(ei);     e = layers.BatchNormalization()(e); e = layers.Activation('relu')(e); e = layers.Dropout(0.2)(e)
e   = layers.Dense(128)(e);      e = layers.BatchNormalization()(e); e = layers.Activation('relu')(e); e = layers.Dropout(0.2)(e)
emb = layers.Dense(enc_dim, activation='linear')(e)
encoder = Model(ei, emb, name='encoder')

di  = keras.Input(shape=(enc_dim,))
d   = layers.Dense(128, activation='relu')(di)
d   = layers.Dense(256, activation='relu')(d)
rec = layers.Dense(in_dim, activation='linear')(d)
decoder = Model(di, rec, name='decoder')

ai          = keras.Input(shape=(in_dim,))
autoencoder = Model(ai, decoder(encoder(ai)))


class ClusteringLayer(layers.Layer):
    def __init__(self, n_clusters, embedding_dim, alpha=1.0, **kwargs):
        super().__init__(**kwargs)
        self.n_clusters    = n_clusters
        self.embedding_dim = embedding_dim
        self.alpha         = alpha

    def build(self, s):
        self.clusters = self.add_weight(
            'clusters', (self.n_clusters, self.embedding_dim),
            initializer='glorot_uniform', trainable=True
        )
        super().build(s)

    def call(self, x):
        z  = tf.expand_dims(x, 1)
        mu = tf.expand_dims(self.clusters, 0)
        d  = tf.reduce_sum(tf.square(z - mu), axis=2)
        q  = 1.0 / (1.0 + d / self.alpha)
        q  = q ** ((self.alpha + 1.0) / 2.0)
        return q / tf.reduce_sum(q, axis=1, keepdims=True)

    def get_config(self):
        c = super().get_config()
        c.update({
            'n_clusters':    self.n_clusters,
            'embedding_dim': self.embedding_dim,
            'alpha':         self.alpha,
        })
        return c


clust_out = ClusteringLayer(K, enc_dim, name='clustering')(encoder(ai))
model     = Model(ai, [clust_out, decoder(encoder(ai))])


print('\n' + '=' * 60 + '\nPHASE 1: AUTOENCODER PRETRAINING\n' + '=' * 60)
autoencoder.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mse')
autoencoder.fit(X, X, epochs=30, batch_size=BS, validation_split=0.1, verbose=1, shuffle=True)
safe_save(encoder, os.path.join(model_dir, 'pretrained_encoder.keras'))


print('\n' + '=' * 60 + '\nPERFORMING SMOTE BALANCING\n' + '=' * 60)
le     = LabelEncoder()
y_int  = le.fit_transform(y_str)
counts = Counter(y_int)
print(f"Before SMOTE: {counts}")

target   = min(max(counts.values()), 10000)
sampling = {cls: max(cnt, target) for cls, cnt in counts.items()}

try:
    smote        = SMOTE(
        sampling_strategy=sampling,
        random_state=42,
        k_neighbors=min(3, min(counts.values()) - 1)
    )
    X_bal, y_bal = smote.fit_resample(X, y_int)
    print(f"After SMOTE: {Counter(y_bal)}")
except Exception as ex:
    print(f"SMOTE failed ({ex}), proceeding unbalanced...")
    X_bal, y_bal = X, y_int


print('\n' + '=' * 60 + '\nPHASE 2: DEEP K-MEANS (K=25)\n' + '=' * 60)
embs = encoder.predict(X_bal, verbose=0)
km   = KMeans(n_clusters=K, n_init=20, random_state=42).fit(embs)
model.get_layer('clustering').set_weights([km.cluster_centers_])

steps = (len(X_bal) // BS) * max_epochs
lr    = keras.optimizers.schedules.CosineDecay(1e-3, steps, alpha=1e-5)
model.compile(
    optimizer=keras.optimizers.Adam(lr),
    loss=['kld', 'mse'],
    loss_weights=[1.0, 0.5]
)

try:
    from swarmlearning.tf import SwarmCallback
    swCb = SwarmCallback(syncFrequency=128, minPeers=min_peers, useAdaptiveSync=False)
    swCb.logger.setLevel(logging.DEBUG)
    cbs  = [swCb]
    print('SwarmCallback OK')
except Exception as ex:
    cbs = []
    print(f'Local mode: {ex}')

print(f'\nTraining Phase 2 ({max_epochs} epochs)...\n')
best_nmi = 0.0

for ep in range(1, max_epochs + 1):
    q, _  = model.predict(X_bal, verbose=0)
    p     = target_dist(q)
    h     = model.fit(X_bal, [p, X_bal], epochs=1, batch_size=BS, verbose=0, callbacks=cbs)

    q_eval, _ = model.predict(X, verbose=0)
    pr        = np.argmax(q_eval, axis=1)
    nmi       = normalized_mutual_info_score(y_int, pr)

    if nmi > best_nmi:
        best_nmi = nmi
        safe_save(model, os.path.join(model_dir, 'best_model.keras'))

    if ep % 10 == 0 or ep == 1:
        dist = ' | '.join([f'C{c}:{n}' for c, n in zip(*np.unique(pr, return_counts=True))])
        print(f'  Ep {ep:3d}/{max_epochs} | Loss:{h.history["loss"][0]:.4f} | NMI:{nmi:.4f} \n      {dist}')

safe_save(model, os.path.join(model_dir, 'final_model.keras'))


q_f, _      = model.predict(X, verbose=0)
final_preds = np.argmax(q_f, axis=1)
cluster_map = {}

for i in range(K):
    idx = np.where(final_preds == i)[0]
    if len(idx) > 0:
        majority_label = Counter(y_str[idx]).most_common(1)[0][0]
        cluster_map[i] = majority_label
    else:
        cluster_map[i] = 'normal'

print(f'\nFinal Cluster Map (Majority Voting):')
for k, v in cluster_map.items():
    print(f"  C{k} -> {v.upper()}")

safe_dump(cluster_map, os.path.join(model_dir, 'cluster_map.pkl'))
print('=' * 60 + '\nALL DONE\n' + '=' * 60)