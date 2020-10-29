#!/bin/bash

# If you are using the local virtual environment setup that is dictated in the
# inital readme, you need to make sure that the awscli has all of the correct
# requirements. This can be accomplished via running in the terminal
# $ aws configure
# and supplying the required credentials
# aws s3 sync data s3://beholder-data
aws s3 sync s3://beholder-data data