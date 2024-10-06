<!-- Insall dedectron2 -->
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git' 

uvicorn segmentation.segmentation:app --host 0.0.0.0 --port 5000