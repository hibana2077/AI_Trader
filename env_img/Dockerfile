FROM ubuntu:22.10

RUN apt-get update
RUN apt-get install curl vim cmake libopenmpi-dev python3-dev zlib1g-dev libgl1-mesa-glx swig git time -y

RUN apt install python3-pip python3-venv -y
RUN pip3 install --upgrade pip

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt
RUN pip install git+https://github.com/AI4Finance-Foundation/FinRL.git

COPY . /work
WORKDIR /work
RUN git clone https://github.com/AI4Finance-Foundation/FinRL.git

EXPOSE 5000
EXPOSE 80
EXPOSE 443

CMD ["tail", "-f", "/dev/null"]