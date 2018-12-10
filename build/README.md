# Building the server image on ubuntu

To build the gws docker image, you need a clone of the gws server repository, and the following packages: 

- python 3.6
- curl
- docker-ce

```
sudo apt-get install python3.6 curl docker-ce
```

(see https://docs.docker.com/install/linux/docker-ce/debian for other docker installation options).

Create a temporary build directory somewhere on your disk: 

```
mkdir gws-build
``` 

From within the build directory, run the build script:

```
cd gws-build
python3 /path/to/gws-server/build/build.py
```

If needed, specify the build `mode` ('debug' or 'release') and a custom image name, for example:

```
python3 /path/to/gws-server/build/build.py debug my-gws-image
```

(on some setups, you might need to invoke it with `sudo`).

The script downloads all neccessary assets and builds the image. Once the image has been built, you can remove the build directory.
