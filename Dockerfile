FROM ghcr.io/scientificcomputing/fenics-gmsh:2023-08-16

COPY . /app
WORKDIR /app
# exposing default port for streamlit
EXPOSE 8501

RUN python3 -m pip install ".[gui]"
