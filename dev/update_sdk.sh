#!/bin/bash

# Exit on error
set -e

# Get latest dev versions from PyPI
echo "Finding latest dev versions..."

# Debug: Print all available versions
echo "Available synth-sdk versions:"
curl -s https://pypi.org/pypi/synth-sdk/json | jq -r '.releases | keys[]'

# More robust version extraction for synth-sdk
SYNTH_SDK_VERSION=$(curl -s https://pypi.org/pypi/synth-sdk/json | \
    jq -r '.releases | keys[]' | \
    grep -E '^\d+\.\d+\.\d+\.dev\d+$' | \
    sort -V | \
    tail -n1)

# Check if version was found
if [ -z "$SYNTH_SDK_VERSION" ]; then
    echo "Error: Could not find dev version"
    echo "synth-sdk version not found"
    exit 1
fi

echo "Installing synth-sdk==${SYNTH_SDK_VERSION}"
uv add synth-sdk==${SYNTH_SDK_VERSION} --refresh

echo "Installation complete!"
