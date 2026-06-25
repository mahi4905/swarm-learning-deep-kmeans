import os
import numpy as np
import pandas as pd


data_dir = os.getenv('DATA_DIR', '/platform/swarmml/data')

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
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty'
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


df = pd.read_csv(os.path.join(data_dir, 'KDDTrain+.txt'), names=col_names, low_memory=False)
df['label'] = df['label'].map(attack_map).fillna('unknown')
df = df.drop(['difficulty'], axis=1)

# Generate basic unbalanced splits (SMOTE will balance them later inside nsl_kdd.py)
grps = {c: df[df['label'] == c] for c in ['normal', 'dos', 'probe', 'r2l', 'u2r']}


def make_node(dom, others, ratio=0.65, size=20000):
    dn = int(size * ratio)
    on = size - dn
    ds = grps[dom].sample(n=dn, replace=len(grps[dom]) < dn, random_state=42)
    per = on // len(others)
    pts = [
        grps[o].sample(n=per, replace=len(grps[o]) < per, random_state=42)
        for o in others
    ]
    return pd.concat([ds] + pts).sample(frac=1, random_state=42).reset_index(drop=True)


configs = [
    ('normal', ['dos', 'probe', 'r2l', 'u2r']),
    ('dos',    ['normal', 'probe', 'r2l', 'u2r']),
    ('probe',  ['normal', 'dos', 'r2l', 'u2r']),
    ('r2l',    ['normal', 'dos', 'probe', 'u2r']),
    ('u2r',    ['normal', 'dos', 'probe', 'r2l']),
]

for i, (dom, others) in enumerate(configs, 1):
    node = make_node(dom, others)
    out = os.path.join(data_dir, f'node{i}')
    os.makedirs(out, exist_ok=True)
    node.to_csv(os.path.join(out, 'KDDTrain+.txt'), index=False)
    print(f"Node {i} created.")

print("Done!")