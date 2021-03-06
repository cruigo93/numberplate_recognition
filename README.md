# Efficientnet b0 based CRNN

pretrained model and encoder [link](https://drive.google.com/drive/folders/1Tfu2G2RKLFSXOIPrKkLhipZ_YZNNiyuL?usp=sharing)

```dockerfile
docker build -t number/deploy .
docker run -it --gpus all -p 9090:9090 -d number/deploy
```

To predict

```json
{
    "base64": "image_in_base64"
}
```
