#!/bin/bash
podman build -f Dockerfile -t single-ancilla-block-encoding-env

podman run -dt \
        --device=/dev/kfd \
        --device=/dev/dri \
        --group-add=video \
        --name single-ancilla-block-encoding \
        -v $(pwd):/workspace \
        -p 8888:8888 \
        -p 2222:22 \
        localhost/single-ancilla-block-encoding-env:latest \
        bash -c "mkdir -p /run/sshd && /usr/sbin/sshd && sleep infinity"
        
cat ~/.ssh/id_ed25519.pub | podman exec -i single-ancilla-block-encoding bash -c "mkdir -p /root/.ssh && chmod 700 /root/.ssh && cat >> /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys"
