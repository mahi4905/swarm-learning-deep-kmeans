#!/bin/bash
set -e
cd ~/swarm-learning
HOST_IP="10.160.0.2"
WS="workspace/nsl-kdd-v6"
IMAGE="nsl-kdd-v6-ml-env"
MAX_EPOCHS="${MAX_EPOCHS:-50}"
MIN_PEERS=5

echo "[1/6] Building image $IMAGE..."
cp $WS/model/nsl_kdd.py $WS/ml-context/
docker build -t $IMAGE $WS/ml-context/

echo "[2/6] Creating node splits..."
docker run --rm -v $(pwd)/$WS/data:/platform/swarmml/data -v $(pwd)/$WS/model:/model -e DATA_DIR=/platform/swarmml/data $IMAGE python3 /model/prepare_data.py

echo "[3/6] Cleaning old containers..."
docker rm -f sn1 ml1 sl1 ml2 sl2 ml3 sl3 ml4 sl4 ml5 sl5 2>/dev/null || true
./scripts/bin/stop-swarm 2>/dev/null || true
sleep 5

echo "[4/6] Starting SN1..."
./scripts/bin/run-sn --name=sn1 --host-ip=$HOST_IP --sentinel --sn-api-port=30304 --key=$WS/cert/sn-1-key.pem --cert=$WS/cert/sn-1-cert.pem --capath=$WS/cert/ca/capath --apls-ip=$HOST_IP &
sleep 15

echo "[5/6] Launching 5 ML/SL nodes (MAX_EPOCHS=$MAX_EPOCHS)..."
for i in 1 2 3 4 5; do
  FS_PORT=$((15000+i*1000))
  screen -dmS "node$i" bash -c "cd ~/swarm-learning && ./scripts/bin/run-sl --name=sl$i --host-ip=$HOST_IP --sn-ip=$HOST_IP --sn-api-port=30304 --sl-fs-port=$FS_PORT --key=$WS/cert/sl-$i-key.pem --cert=$WS/cert/sl-$i-cert.pem --capath=$WS/cert/ca/capath --ml-image=$IMAGE --ml-name=ml$i --ml-w=/tmp/nsl-kdd --ml-entrypoint=python3 --ml-cmd=model/nsl_kdd.py --ml-v=\$(pwd)/$WS/model:/tmp/nsl-kdd/model --ml-v=\$(pwd)/$WS/data/node$i:/tmp/nsl-kdd/data --ml-e DATA_DIR=/tmp/nsl-kdd/data --ml-e MODEL_DIR=/tmp/nsl-kdd/model --ml-e MAX_EPOCHS=$MAX_EPOCHS --ml-e MIN_PEERS=$MIN_PEERS --apls-ip=$HOST_IP"
done

echo "[6/6] All nodes launched!"
while true; do
  clear
  echo "===== NSL-KDD v4 OVER-CLUSTERED | $(date) ====="
  for i in 1 2 3 4 5; do
    last=$(docker logs ml$i 2>&1 | grep -E "Ep |PHASE|SMOTE" | tail -1)
    printf "ml%-4s %s\n" "$i" "${last:0:80}"
  done
  echo ""
  docker logs ml1 2>&1 | grep -i "merge\|sync" | tail -1
  sleep 30
done
