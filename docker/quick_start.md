## Quick Start

### Install docker compose
[guide](https://docs.docker.com/compose/install/)

### Build and start
```
# build container:
USER_ID=$UID docker-compose build detectron2

# start container stack
USER_ID=$UID docker-compose up -d
```

## Feature

SSH into the container with 
`ssh appuser@localhost -p 6022` with password `password`
The actual password can be changed in the `Dockerfile`

Once you successfully ssh in the container you will be able to config your ssh interpreter easily

## Change dir ownership 
This is a word-around. You may need to change the dir ownership (inside container)
that mounted to the container.

Change ownership if necessary
```bash
# In container
cd ~
sudo chown appuser:sudo detectron2
ls -l
```
[discussion for mount folder with non-root user](https://github.com/moby/moby/issues/2259)


## Mount AWS EBS block if necessary
```bash
sudo chown appuser:sudo detectron2
ls -l
mount volume lsblk
sudo mount /dev/XXXXX /data
```



