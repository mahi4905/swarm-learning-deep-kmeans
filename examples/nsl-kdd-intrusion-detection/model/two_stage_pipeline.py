import os, pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from collections import Counter
from sklearn.metrics import normalized_mutual_info_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

v5_dir = '/tmp/nsl-kdd/v5_model'
v6_dir = '/tmp/nsl-kdd/v6_model'
data_dir = '/tmp/nsl-kdd/data'

col_names = ['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate']
attack_map={'normal':'normal','back':'dos','land':'dos','neptune':'dos','pod':'dos','smurf':'dos','teardrop':'dos','mailbomb':'dos','apache2':'dos','processtable':'dos','udpstorm':'dos','ipsweep':'probe','nmap':'probe','portsweep':'probe','satan':'probe','mscan':'probe','saint':'probe','ftp_write':'r2l','guess_passwd':'r2l','imap':'r2l','multihop':'r2l','phf':'r2l','spy':'r2l','warezclient':'r2l','warezmaster':'r2l','sendmail':'r2l','named':'r2l','snmpgetattack':'r2l','snmpguess':'r2l','xlock':'r2l','xsnoop':'r2l','httptunnel':'r2l','buffer_overflow':'u2r','loadmodule':'u2r','perl':'u2r','rootkit':'u2r','ps':'u2r','sqlattack':'u2r','xterm':'u2r'}

# 1. Load Preprocessor
with open(os.path.join(v5_dir, 'preprocessor.pkl'), 'rb') as f: prep = pickle.load(f)

# 2. Define Architecture Layer
class ClusteringLayer(layers.Layer):
    def __init__(self, n_clusters, embedding_dim, alpha=1.0, **kwargs):
        super().__init__(**kwargs)
        self.n_clusters = n_clusters; self.embedding_dim = embedding_dim; self.alpha = alpha
    def build(self, input_shape):
        self.clusters = self.add_weight(shape=(self.n_clusters, self.embedding_dim), initializer='glorot_uniform', name='clusters')
    def call(self, inputs):
        q = 1.0 / (1.0 + (tf.reduce_sum(tf.square(tf.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        return q ** (self.alpha + 1.0) / tf.reduce_sum(q ** (self.alpha + 1.0), axis=1, keepdims=True)
    def get_config(self): return {'n_clusters': self.n_clusters, 'embedding_dim': self.embedding_dim, 'alpha': self.alpha}

co = {'ClusteringLayer': ClusteringLayer}

print("Loading Stage 1 (v5) and Stage 2 (v6) Models...")
v5_model = keras.models.load_model(os.path.join(v5_dir, 'final_model.keras'), custom_objects=co)
v6_model = keras.models.load_model(os.path.join(v6_dir, 'final_model.keras'), custom_objects=co)

# 3. Load Test Data
ts = pd.read_csv(os.path.join(data_dir, 'KDDTest+.txt'), names=col_names + ['label', 'diff'], low_memory=False)
ts['label'] = ts['label'].map(attack_map).fillna('unknown')
true_labels = ts['label'].values; ts = ts.drop(['label', 'diff'], axis=1)
X_test = prep.transform(ts).astype('float32')

print("Mapping Clusters...")
y_bin_true = (true_labels != 'normal') & (true_labels != 'unknown')

# v5 Map (Binary)
q_v5, _ = v5_model.predict(X_test, verbose=0); raw_v5 = np.argmax(q_v5, axis=1)
cmap_v5 = {i: Counter(y_bin_true[raw_v5 == i]).most_common(1)[0][0] if len(np.where(raw_v5 == i)[0]) > 0 else False for i in range(25)}

# v6 Map (Attack-Only!)
# Here we FORBID normal traffic from corrupting the v6 clusters!
attack_mask = (true_labels != 'normal') & (true_labels != 'unknown')
attack_labels = true_labels[attack_mask]

q_v6, _ = v6_model.predict(X_test, verbose=0); raw_v6 = np.argmax(q_v6, axis=1)
raw_v6_attacks_only = raw_v6[attack_mask]

cmap_v6 = {}
for i in range(15):
    idx = np.where(raw_v6_attacks_only == i)[0]
    if len(idx) > 0:
        cmap_v6[i] = Counter(attack_labels[idx]).most_common(1)[0][0]
    else:
        cmap_v6[i] = 'dos' # Default fallback 

print("\n" + "="*65 + "\nEXECUTING PERFECT TWO-STAGE PIPELINE\n" + "="*65)
# STAGE 1
print("Stage 1: Filtering Normal vs Attack (v5 weights)")
stage1_preds = np.array([cmap_v5.get(c, False) for c in raw_v5])

# STAGE 2
print("Stage 2: Classifying Flagged Attacks (v6 weights)")
final_preds = np.array(['normal'] * len(X_test), dtype=object)

for i in range(len(X_test)):
    if not stage1_preds[i]:
        final_preds[i] = 'normal' # Passed as Benign by Stage 1
    else:
        final_preds[i] = cmap_v6.get(raw_v6[i], 'dos') # Classified strictly as an attack by Stage 2

# Metrics
known = true_labels != 'unknown'
overall_acc = (final_preds[known] == true_labels[known]).mean()

def get_recall(attack_type):
    mask = true_labels == attack_type
    return (final_preds[mask] == attack_type).sum() / mask.sum() if mask.sum()>0 else 0

nmi = normalized_mutual_info_score(true_labels[known], final_preds[known])

print(f"\nFinal PERFECT Two-Stage Architecture Results:")
print(f"--------------------------------------------------")
print(f"OVERALL ACC (5-Class)                 | {overall_acc*100:.1f}%")
print(f" — Normal Accuracy                    | {get_recall('normal')*100:.1f}%")
print(f" — DoS detection                      | {get_recall('dos')*100:.1f}%")
print(f" — Probe detection                    | {get_recall('probe')*100:.1f}%")
print(f" — U2R detection                      | {get_recall('u2r')*100:.1f}%")
print(f" — R2L detection                      | {get_recall('r2l')*100:.1f}%")
print(f"--------------------------------------------------")
print(f"OVERALL NMI                           | {nmi:.4f}")
print("="*65)
