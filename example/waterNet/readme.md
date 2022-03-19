# modify paths
Paths are managed in 
```
hydroDL\kPath.py
```
To run the code on a new device, add a following elif:
```python
elif hostName == {your-host-name}:
    dirWQ = r'{local-dir}\waterQuality'
    dirData = r'{local-dir}\data'
```
# python envirment
check the conda environment file:
```
environment.yml
```
