# Privacy-Preserving Swarm Learning for Network Intrusion Detection

This example demonstrates how a privacy-preserving Network Intrusion Detection System (NIDS) built using the NSL-KDD dataset can be trained and deployed on the HPE Swarm Learning platform.

The objective is to collaboratively detect cyberattacks across multiple organizations without sharing raw network traffic logs. The system uses a Two-Stage Hybrid Deep Learning Architecture consisting of binary attack detection followed by multi-class attack classification.

## Dataset Description

This project uses the **NSL-KDD Dataset**, a benchmark dataset for network intrusion detection.

Dataset Information:

https://www.unb.ca/cic/datasets/nsl.html

The dataset contains network traffic records categorized into:

- Normal
- DoS (Denial of Service)
- Probe
- U2R (User-to-Root)
- R2L (Remote-to-Local)

The dataset files and non-IID node splits are stored in:

```text
nsl-kdd-intrusion-detection/app-data
```

## Data Preprocessing

Raw network traffic contains categorical values such as:

- tcp
- udp
- http
- ftp
- SF

These values are transformed using a custom preprocessing pipeline.

The preprocessing stage:

- Applies One-Hot Encoding to categorical features
- Applies Min-Max Scaling to continuous features
- Generates normalized 122-dimensional feature vectors

The preprocessing artifacts are stored in:

```text
data-prep/preprocessor.pkl
```

## Non-IID Data Distribution

To simulate real-world enterprise environments, the dataset is distributed across five Swarm Learning nodes.

### Node 1 – Public Web Server

Predominantly contains DoS traffic.

### Node 2 – Internal Database Firewall

Predominantly contains Probe traffic.

### Nodes 3, 4 and 5 – Regional Routers

Contain mixed traffic distributions.

This setup reflects realistic deployments where organizations observe different attack patterns and traffic characteristics.

## Model Architecture

The intrusion detection framework consists of two stages.

### Stage 1 – Binary Classification

Classifies network traffic as:

- Normal
- Attack

### Stage 2 – Multi-Class Classification

Further classifies attacks into:

- DoS
- Probe
- U2R
- R2L

The implementation uses:

- TensorFlow / Keras
- Deep K-Means Autoencoder
- HPE Swarm Learning

## Performance

After Swarm Learning training across five distributed nodes:

| Metric | Accuracy |
|----------|----------|
| Binary Classification | 86.6% |
| Multi-Class Classification | 73.4% |

No participant shares raw network logs during training.

## Cluster Setup

The following image illustrates a Swarm Learning cluster setup:

![NSL-KDD Cluster Setup](/docs/User/GUID-PLACEHOLDER.png)

- One Sentinel Node (SN1) runs on host-1.
- Five Swarm Learning nodes participate in collaborative training.
- Each SL node launches an ML node as a sidecar container.
- All nodes communicate through the Sentinel node.
- The License Server runs on host-1 using the default port 5814.
- Training is performed without sharing raw network traffic data.

## Running the NSL-KDD Example

### 1. Navigate to the Swarm Learning Directory

On all participating hosts:

```bash
cd swarm-learning
```

### 2. Create a Workspace

On all hosts:

```bash
mkdir workspace

cp -r examples/nsl-kdd-intrusion-detection workspace/

cp -r examples/utils/gen-cert \
workspace/nsl-kdd-intrusion-detection/
```

### 3. Generate Certificates

On host-1:

```bash
./workspace/nsl-kdd-intrusion-detection/gen-cert \
-e nsl-kdd-intrusion-detection -i 1
```

On host-2:

```bash
./workspace/nsl-kdd-intrusion-detection/gen-cert \
-e nsl-kdd-intrusion-detection -i 2
```

Repeat for additional hosts if using more nodes.

### 4. Exchange CA Certificates

On host-1:

```bash
scp host-2:workspace/nsl-kdd-intrusion-detection/cert/ca/capath/ca-2-cert.pem \
workspace/nsl-kdd-intrusion-detection/cert/ca/capath
```

On host-2:

```bash
scp host-1:workspace/nsl-kdd-intrusion-detection/cert/ca/capath/ca-1-cert.pem \
workspace/nsl-kdd-intrusion-detection/cert/ca/capath
```

### 5. Build the ML Docker Image

Copy the Swarm Learning wheel file:

```bash
cp -L lib/swarmlearning-client-py3-none-manylinux_2_24_x86_64.whl \
workspace/nsl-kdd-intrusion-detection/ml-context/
```

Build the Docker image:

```bash
docker build -t nsl-kdd-ml-env \
workspace/nsl-kdd-intrusion-detection/ml-context
```

### 6. Launch Sentinel Node

On host-1:

```bash
./scripts/bin/run-sn -d \
--name=sn1 \
--host-ip=172.1.1.1 \
--sentinel \
--sn-api-port=30304 \
--key=workspace/nsl-kdd-intrusion-detection/cert/sn-1-key.pem \
--cert=workspace/nsl-kdd-intrusion-detection/cert/sn-1-cert.pem \
--capath=workspace/nsl-kdd-intrusion-detection/cert/ca/capath \
--apls-ip=172.1.1.1
```

The Sentinel node is ready when the following message appears:

```text
swarm.blCnt : INFO : Starting SWARM-API-SERVER on port: 30304
```

### 7. Launch Swarm Learning Nodes

Example command:

```bash
./scripts/bin/run-sl \
--name=sl1 \
--host-ip=172.1.1.1 \
--sn-ip=172.1.1.1 \
--sn-api-port=30304 \
--sl-fs-port=16000 \
--key=workspace/nsl-kdd-intrusion-detection/cert/sl-1-key.pem \
--cert=workspace/nsl-kdd-intrusion-detection/cert/sl-1-cert.pem \
--capath=workspace/nsl-kdd-intrusion-detection/cert/ca/capath \
--ml-it \
--ml-image=nsl-kdd-ml-env \
--ml-name=ml1 \
--ml-w=/tmp/test \
--ml-entrypoint=python3 \
--ml-cmd=model/train_stage1.py \
--ml-v=workspace/nsl-kdd-intrusion-detection/model:/tmp/test/model \
--ml-v=workspace/nsl-kdd-intrusion-detection/data-prep:/tmp/test/data-prep \
--ml-e DATA_DIR=data-prep \
--ml-e MODEL_DIR=model \
--ml-e MAX_EPOCHS=50 \
--ml-e MIN_PEERS=5 \
--apls-ip=172.1.1.1
```

Repeat for all participating nodes.

### 8. Monitor Training

Monitor ML containers using:

```bash
docker logs -f ml1
```

Training completes successfully when the following message appears:

```text
SwarmCallback : INFO : Saved the trained model - model/final_model.keras
```

### 9. Run the Two-Stage Inference Pipeline

After training:

```bash
python3 model/two_stage_pipeline.py
```

The pipeline:

1. Detects Normal vs Attack traffic
2. Classifies detected attacks into DoS, Probe, U2R, or R2L

### 10. Cleanup

On all hosts:

```bash
./scripts/bin/stop-swarm
```

This stops and removes all Sentinel, Swarm Learning, and ML containers.

