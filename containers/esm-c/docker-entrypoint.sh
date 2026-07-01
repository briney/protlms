#!/bin/bash
# Entrypoint shim that creates a /etc/passwd-compatible entry for arbitrary
# UIDs so that getpass.getuser() / getpwuid() works even when the container
# runs as a host user not present in /etc/passwd.
#
# Background: torch._dynamo (imported transitively by transformers, whose
# generation utils import torch._dynamo.package at module load) computes a
# cache directory via getpass.getuser() at import time. When running with
# --user uid:gid and the uid is not in the container's /etc/passwd,
# getpwuid() raises KeyError. libnss_wrapper shims the NSS layer to return a
# synthetic entry for the current uid.

set -e

if [ -z "$(getent passwd "$(id -u)" 2>/dev/null)" ]; then
    NSS_PASSWD_COPY="$(mktemp)"
    cat /etc/passwd > "${NSS_PASSWD_COPY}"
    echo "user:x:$(id -u):$(id -g):container user:/tmp:/bin/false" >> "${NSS_PASSWD_COPY}"
    export NSS_WRAPPER_PASSWD="${NSS_PASSWD_COPY}"
    export NSS_WRAPPER_GROUP=/etc/group
    export LD_PRELOAD=libnss_wrapper.so
fi

exec python /app/entrypoint.py "$@"
