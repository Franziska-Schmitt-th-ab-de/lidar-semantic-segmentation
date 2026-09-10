#!/bin/bash

# Detect the operating system
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    user_id=$(id -u)
    user_name=$(whoami)
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # For Windows, set defaults or handle specific cases
    user_id=1000
    user_name=appuser
else
    echo "Unsupported OS: $OSTYPE"
    exit 1
fi

# Write the user name and ID to the .env file
echo "USER_ID=$user_id" > .env
echo "USER_NAME=$user_name" >> .env

echo "Environment variables written to .env file:"
echo "USER_NAME=$user_name"
echo "USER_ID=$user_id"
