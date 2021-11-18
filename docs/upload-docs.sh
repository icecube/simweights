#!/bin/bash

GIT_BRANCH=`git rev-parse --abbrev-ref HEAD`
GIT_REVISION=`git rev-parse --verify --short HEAD`
TARBALL_NAME="simweights-docs-${GIT_BRANCH}-${GIT_REVISION}.tar.gz"

if [ -z "${ICECUBE_PASSWORD}" ]; then
    echo ICECUBE_PASSWORD not set, skipping upload
    exit 0
fi

if [ "${GIT_BRANCH}" != "main" ]; then
    echo not on main branch, skipping upload
    exit 0
fi

TARCMD="tar -czvf $TARBALL_NAME -C_build/html ."
echo running $TARCMD
${TARCMD}

UPLOADCMD="curl -XPUT -i --data-binary @${TARBALL_NAME} https://docs.icecube.aq/api/upload?path=simweights/main -u icecube:${ICECUBE_PASSWORD}"
echo running ${UPLOADCMD}
${UPLOADCMD}
exit $?
