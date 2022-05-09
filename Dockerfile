FROM finsberg/fenics:latest

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install h5py --no-binary=h5py

COPY . /app
WORKDIR /app
# exposing default port for streamlit
EXPOSE 8501

RUN python3 -m pip install ".[gui]"
