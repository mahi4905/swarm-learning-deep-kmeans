import os
import pandas as pd

DATA_DIR = os.getenv("DATA_DIR", "/platform/swarmml/data")

COL_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty"
]

ATTACK_MAP = {attack: "attack" for attack in [
    "back", "land", "neptune", "pod", "smurf", "teardrop", "mailbomb",
    "apache2", "processtable", "udpstorm", "ipsweep", "nmap", "portsweep",
    "satan", "mscan", "saint", "ftp_write", "guess_passwd", "imap",
    "multihop", "phf", "spy", "warezclient", "warezmaster", "sendmail",
    "named", "snmpgetattack", "snmpguess", "xlock", "xsnoop",
    "httptunnel", "buffer_overflow", "loadmodule", "perl", "rootkit",
    "ps", "sqlattack", "xterm"
]}
ATTACK_MAP["normal"] = "normal"

print("Loading full training data...")

df = pd.read_csv(
    os.path.join(DATA_DIR, "KDDTrain+.txt"),
    names=COL_NAMES,
    low_memory=False
)

df["label"] = df["label"].map(ATTACK_MAP).fillna("attack")
df = (
    df[df["label"].isin(["normal", "attack"])]
    .drop(columns="difficulty")
    .reset_index(drop=True)
)

print(f"Total rows: {len(df)}")
print(df["label"].value_counts())

normal_df = df.query("label == 'normal'")
attack_df = df.query("label == 'attack'")

configs = [
    (13000, 7000, 1),
    (7000, 13000, 2),
    (10000, 10000, 3),
]

for idx, (normal_n, attack_n, seed) in enumerate(configs, start=1):

    node = pd.concat([
        normal_df.sample(normal_n, random_state=seed),
        attack_df.sample(attack_n, random_state=seed)
    ]).sample(frac=1, random_state=seed).reset_index(drop=True)

    out_dir = os.path.join(DATA_DIR, f"node{idx}")
    os.makedirs(out_dir, exist_ok=True)

    node.to_csv(os.path.join(out_dir, "KDDTrain+.txt"), index=False)

    print(
        f"Node {idx}: {len(node)} rows | "
        f"{node['label'].value_counts().to_dict()}"
    )

print("Done!")