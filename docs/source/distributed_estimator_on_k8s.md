# Train DeepCTR Estimator using ElasticDL on Kubernetes

This document shows how to run an ElasticDL job to train a deepctr
estimator model (DeepFM) using criteo dataset on Kubernetes.

## Prerequisites

1. Install Minikube, preferably >= v1.11.0, following the installation
   [guide](https://kubernetes.io/docs/tasks/tools/install-minikube).  Minikube
   runs a single-node Kubernetes cluster in a virtual machine on your personal
   computer.

1. Install Docker CE, preferably >= 18.x, following the
   [guide](https://docs.docker.com/docker-for-mac/install/) for building Docker
   images containing user-defined models and the ElasticDL framework.

1. Install Python, preferably >= 3.6, because the ElasticDL command-line tool is
   in Python.

## Models

In this tutorial, we use a [DeepFM estimator](https://github.com/shenweichen/DeepCTR/blob/master/deepctr/estimator/models/deepfm.py)
model in DeepCTR. The complete program to train the model with dataset definition is in [ElasticDL
model zoo]().

## Datasets

In this tutorial, We use the criteo dataset in DeepCTR examples.

```bash
mkdir ./data
wget https://github.com/shenweichen/DeepCTR/blob/master/examples/criteo_sample.txt -O ./data/iris.data
```

## The Kubernetes Cluster

The following command starts a Kubernetes cluster locally using Minikube.  It
uses [VirtualBox](https://www.virtualbox.org/), a hypervisor coming with
macOS, to create the virtual machine cluster.

```bash
minikube start --vm-driver=virtualbox \
  --cpus 2 --memory 6144 --disk-size=50gb 
eval $(minikube docker-env)
```

The command `minikube docker-env` returns a set of Bash environment variable
exports to configure your local environment to re-use the Docker daemon inside
the Minikube instance.

The following command is necessary to enable
[RBAC](https://kubernetes.io/docs/reference/access-authn-authz/rbac/) of
Kubernetes.

```bash
kubectl apply -f \
  https://raw.githubusercontent.com/sql-machine-learning/elasticdl/develop/elasticdl/manifests/elasticdl-rbac.yaml
```

If you happen to live in a region where `raw.githubusercontent.com` is banned,
you might want to Git clone the above repository to get the YAML file.

## Install ElasticDL Client Tool

The following command installs the command line tool `elasticdl`, which talks to
the Kubernetes cluster and operates ElasticDL jobs.

```bash
pip install elasticdl_client
```

## Build the Docker Image with Model Definition

Kubernetes runs Docker containers, so we need to put the training system,
consisting of user-defined models, ElasticDL the trainer, and all dependencies,
into a Docker image.

In this tutorial, we use a predefined model in the ElasticDL repository.  To
retrieve the source code, please run the following command.

```bash
git clone https://github.com/sql-machine-learning/elasticdl
```

Model definitions are in directory `elasticdl/model_zoo/deepctr/`.

### Build the Docker Image for TensorFlow PS and Worker

We build the image based on tensorflow:1.13.2 and the dockerfile
is

```text
FROM tensorflow/tensorflow:1.13.2-py3 as base

RUN pip install elasticdl_api
RUN pip install deepctr

COPY ./model_zoo model_zoo
ENV PYTHONUNBUFFERED 0
```

Then, we use docker to build the image

```bash
docker build -t elasticdl:deepctr_estimator -f ${iris_dockerfile} .
```

## Submit the Training Job

The following command submits a training job:

```bash
elasticdl train \
  --image_name=elasticdl/elasticdl:1.0.0 \
  --worker_image=elasticdl:deepctr_estimator \
  --ps_image=elasticdl:deepctr_estimator \
  --job_command="python -m model_zoo.deepctr.deepfm_estimator --training_data=/data/criteo_sample.txt --validation_data=/data/criteo_sample.txt" \
  --num_workers=1 \
  --num_ps=1 \
  --num_evaluator=1 \
  --master_resource_request="cpu=0.2,memory=1024Mi" \
  --master_resource_limit="cpu=1,memory=2048Mi" \
  --ps_resource_request="cpu=0.2,memory=1024Mi" \
  --ps_resource_limit="cpu=1,memory=2048Mi" \
  --worker_resource_request="cpu=0.3,memory=1024Mi" \
  --worker_resource_limit="cpu=1,memory=2048Mi" \
  --chief_resource_request="cpu=0.3,memory=1024Mi" \
  --chief_resource_limit="cpu=1,memory=2048Mi" \
  --evaluator_resource_request="cpu=0.3,memory=1024Mi" \
  --evaluator_resource_limit="cpu=1,memory=2048Mi" \
  --job_name=test-deepfm-estimator \
  --image_pull_policy=Never \
  --distribution_strategy=ParameterServerStrategy \
  --need_tf_config=true \
  --volume="host_path={criteo_data_path},mount_path=/data" \
```

`{criteo_data_path}` is the absolute path of the `./data` with `criteo_sample.txt`.
Here, the option `--volume="host_path={criteo_data_path},mount_path=/data"`
bind mount it into the containers/pods.

The option `--num_workers=1` tells the master container to start a worker pod.
The option `--num_ps=1` tells the master container to start a ps pod.
The option `--num_evaluator=1` tells the master container to start an evaluator pod.

And the master will start a chief worker for a TensorFlow estimator model by default.

### Check Job Status

After the job submission, we can run the command `kubectl get pods` to list
related containers.

```bash
NAME                                     READY   STATUS    RESTARTS   AGE
elasticdl-test-deepctr-estimator-master     1/1     Running   0          9s
test-deepctr-estimator-edljob-chief-0       1/1     Running   0          6s
test-deepctr-estimator-edljob-evaluator-0   0/1     Pending   0          6s
test-deepctr-estimator-edljob-ps-0          1/1     Running   0          7s
test-deepctr-estimator-edljob-worker-0      1/1     Running   0          6s
```

We can view the log of workers by `kubectl logs test-deepctr-estimator-edljob-chief-0 `.

```text
INFO:tensorflow:global_step/sec: 4.84156
INFO:tensorflow:global_step/sec: 4.84156
INFO:tensorflow:Saving checkpoints for 203 into /data/ckpts/model.ckpt.
INFO:tensorflow:Saving checkpoints for 203 into /data/ckpts/model.ckpt.
INFO:tensorflow:global_step/sec: 7.05433
INFO:tensorflow:global_step/sec: 7.05433
```

## Train an Estimator Using ElasticDL with Your Dataset

You only need to modify your `input_fn` with ElasticDL DataShardService.
The DataShardService will split the sample indices into ranges and assign
those ranges to workers. The worker only need to read samples by indices
in those ranges.

1. Create a DataShardService.

```python
from elasticai_api.common.data_shard_service import DataShardService

training_data_shard_svc = DataShardService(
        batch_size=batch_size,
        num_epochs=100,
        dataset_size=len(rows),
        num_minibatches_per_shard=1,
        dataset_name="training_data",
    )
```

- batch_size: Batch size of each step.
- num_epochs: The number of epochs.
- dataset_size: The total number of samples in the dataset.
- num_minibatches_per_shard: The number of batches in each shard.
  The number of samples in each shard is
  `batch_size * num_minibatches_per_shard`
- dataset_name: The name of dataset.

2. Create a generator by reading samples with shards.

The `shard.start` and `shard.end` is the start index
and end index of samples in those shard. You can read
samples by the two indices like:

```python
def train_generator(shard_service):
    while True:
        # Read samples by the range of the shard from
        # the data shard serice.
        shard = shard_service.fetch_shard()
        if not shard:
            break
        for i in range(shard.start, shard.end):
            values = read_data_by_index(i)
            yield values
```

3. Create a session hook to report shard

```python
class ElasticDataShardReportHook(tf.train.SessionRunHook):
    def __init__(self, data_shard_service) -> None:
        self._data_shard_service = data_shard_service

    def after_run(self, run_context, run_values):
        try:
            self._data_shard_service.report_batch_done()
        except Exception as ex:
            logging.info("elastic_ai: report batch done failed: %s", ex)
```

After 3 steps, you can train your estimator models using ElasticDL
in data-parallel.
