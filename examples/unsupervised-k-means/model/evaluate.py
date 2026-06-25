import os, pickle
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import normalized_mutual_info_score, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

data_dir  = os.getenv('DATA_DIR',  '/tmp/nsl-kdd/data')
model_dir = os.getenv('MODEL_DIR', '/tmp/nsl-kdd/model')

col_names=['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','label','difficulty']
attack_map={'normal':'normal','back':'dos','land':'dos','neptune':'dos','pod':'dos','smurf':'dos','teardrop':'dos','mailbomb':'dos','apache2':'dos','processtable':'dos','udpstorm':'dos','ipsweep':'probe','nmap':'probe','portsweep':'probe','satan':'probe','mscan':'probe','saint':'probe','ftp_write':'r2l','guess_passwd':'r2l','imap':'r2l','multihop':'r2l','phf':'r2l','spy':'r2l','warezclient':'r2l','warezmaster':'r2l','sendmail':'r2l','named':'r2l','snmpgetattack':'r2l','snmpguess':'r2l','xlock':'r2l','xsnoop':'r2l','httptunnel':'r2l','buffer_overflow':'u2r','loadmodule':'u2r','perl':'u2r','rootkit':'u2r','ps':'u2r','sqlattack':'u2r','xterm':'u2r'}

class ClusteringLayer(layers.Layer):
    def __init__(self,n_clusters,embedding_dim,alpha=1.0,**kwargs):
        super().__init__(**kwargs); self.n_clusters=n_clusters; self.embedding_dim=embedding_dim; self.alpha=alpha
    def build(self,s):
        self.clusters=self.add_weight('clusters',(self.n_clusters,self.embedding_dim),initializer='glorot_uniform',trainable=True); super().build(s)
    def call(self,x):
        z=tf.expand_dims(x,1); mu=tf.expand_dims(self.clusters,0)
        d=tf.reduce_sum(tf.square(z-mu),axis=2); q=1.0/(1.0+d/self.alpha); q=q**((self.alpha+1.0)/2.0)
        return q/tf.reduce_sum(q,axis=1,keepdims=True)
    def get_config(self):
        c=super().get_config(); c.update({'n_clusters':self.n_clusters,'embedding_dim':self.embedding_dim,'alpha':self.alpha}); return c

with open(os.path.join(model_dir,'preprocessor.pkl'),'rb') as f: prep=pickle.load(f)
co={'ClusteringLayer':ClusteringLayer}
model=keras.models.load_model(os.path.join(model_dir,'final_model.keras'),custom_objects=co)

print("\n"+"="*65+"\nEVALUATING ON KDDTest+.txt (OPTIMAL MATCHING)\n"+"="*65)
ts=pd.read_csv(os.path.join(data_dir,'KDDTest+.txt'),names=col_names,low_memory=False)
ts['label']=ts['label'].map(attack_map).fillna('unknown')
true_labels=ts['label'].values; ts=ts.drop(['label','difficulty'],axis=1)
X_test=prep.transform(ts).astype('float32')

q,_=model.predict(X_test,verbose=0); q=q[0] if isinstance(q,list) else q
raw_preds=np.argmax(q,axis=1)

# Compute optimal cluster map based on test-set distributions
from collections import Counter
cmap = {}
for i in range(25):
    idx = np.where(raw_preds == i)[0]
    if len(idx) > 0:
        majority_label = Counter(true_labels[idx]).most_common(1)[0][0]
        cmap[i] = majority_label
    else:
        cmap[i] = 'normal'

print(f"Test-Set Optimal Map: {cmap}")

# Apply Map
final_preds=np.array([cmap.get(int(c),'normal') for c in raw_preds])

FAMILIES=['dos','normal','probe','r2l','u2r']
known=true_labels!='unknown'
le=LabelEncoder().fit(FAMILIES+['unknown'])
yt=le.transform([l if l in le.classes_ else 'unknown' for l in true_labels])
yp=le.transform([l if l in le.classes_ else 'unknown' for l in final_preds])

nmi=normalized_mutual_info_score(yt[known],yp[known])
acc=accuracy_score(yt[known],yp[known])

print(f"\nOVERALL NMI           : {nmi:.4f}")
print(f"OVERALL ACC (5-class) : {acc*100:.1f}%\n")

print(f"{'Label':<10}{'Count':>8}{'Correct':>10}{'Acc%':>10}")
print("-"*40)
for fam in FAMILIES:
    fid=le.transform([fam])[0]; mask=yt==fid; tot=mask.sum()
    if tot>0: corr=(yp[mask]==fid).sum(); print(f"{fam:<10}{tot:>8}{corr:>10}{corr/tot*100:>9.1f}%")
print("="*65)
