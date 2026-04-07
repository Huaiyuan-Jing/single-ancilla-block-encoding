FROM docker.io/rocm/pytorch:latest

WORKDIR /workspace

RUN apt-get update && apt-get install -y openssh-server && \
    mkdir -p /var/run/sshd && \
    echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config && \
    echo 'PasswordAuthentication yes' >> /etc/ssh/sshd_config

EXPOSE 22

RUN pip install --upgrade pip

RUN pip install pennylane pennylane-lightning[kokkos]

RUN pip install qiskit qiskit-aer

RUN pip install matplotlib scipy jupyterlab

RUN pip install ipykernel
