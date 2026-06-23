# Data

```base
/opt/conda/envs/torch-base/bin/python -c "from modelscope.hub.api import HubApi; HubApi().login('YOUR_TOKEN'); from modelscope import snapshot_download; snapshot_download('CrazyLewis/SHINE_SWE_OPENSOURCE', repo_type='dataset', local_dir='./data/SHINE_SWE_OPENSOURCE')"
```