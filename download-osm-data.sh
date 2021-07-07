#!/bin/bash

cd sloop/datasets

fileid="1ccpGPsXYkJXLFnFLKrWxf7-cElDdZF-u"
filename="SLM-OSM-Dataset.tar.gz"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}
rm -rf cookie

tar xzvf $filename
rm -rf $filename
