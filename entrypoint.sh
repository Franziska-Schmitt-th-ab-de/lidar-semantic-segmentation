#!/bin/bash
echo "--- Setting up container, please wait..."
echo "--- Now as root..."

# Run commands as root using sudo
sudo service ssh start
sudo setfacl -R -d -m u::rwx,g::rwx,o::rwx /workspace
sudo setfacl -R -m u::rwx,g::rwx,o::rwx /workspace

echo "--- End of root..."
echo "--- Additional container setup completed, ready for work..."
tail -F /dev/null
