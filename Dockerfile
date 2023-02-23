FROM ghcr.io/scientificcomputing/fenics-gmsh:2023-02-20

COPY . /app
WORKDIR /app
# exposing default port for streamlit
EXPOSE 8501

RUN python3 -m pip install ".[gui]"
