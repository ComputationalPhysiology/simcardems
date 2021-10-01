FROM quay.io/fenicsproject/stable:latest

COPY . /app
WORKDIR /app

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install h5py --no-binary=h5py && \
    python3 -m pip install "."

# Something wrong with matplotlib fonts
RUN sudo apt install msttcorefonts -qq && rm ~/.cache/matplotlib -rf


ENTRYPOINT ["python3", "-m", "simcardems"]
