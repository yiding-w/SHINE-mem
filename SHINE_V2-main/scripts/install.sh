source /apdcephfs_zwfy/share_303937731/xiyuanwang/liuyewei/export.sh

pip install huggingface==0.0.1 modelscope==1.31.0 transformers==5.5.4 datasets==4.4.1 scikit-learn==1.7.2 hydra-core==1.3.2 wandb openai==2.6.1 rouge==1.0.1 seaborn==0.13.2 matplotlib==3.10.7 multiprocess==0.70.16 paramiko flash-linear-attention accelerate wandb

CAUSAL_CONV1D_FORCE_BUILD=TRUE pip install causal_conv1d --no-build-isolation 

pip install --upgrade tilelang

pip install liger-kernel

pip install ring-flash-attn 

yum install -y libibverbs libibverbs-devel rdma-core
# or
# apt-get install -y libibverbs-dev rdma-core 