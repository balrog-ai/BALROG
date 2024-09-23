docker build -t icl-bench \
    --build-arg UID=$(id -u) \
    --build-arg GID=$(id -g) \
    --build-arg GIT_ACCESS_TOKEN=$GIT_ACCESS_TOKEN \
    -f ./Dockerfile \
    ..