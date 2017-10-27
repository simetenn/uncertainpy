if [ -z "$1" ]; then
    echo "ERROR: No channel provided"
    echo "Usage: upload.sh <channel> [label]"
    exit
fi
if [ $TRAVIS_TEST_RESULT -eq 0 ]; then
    echo "Package name $PACKAGE"
    conda convert "$PACKAGE" --platform win-64 -o packages
    conda convert "$PACKAGE" --platform osx-64 -o packages
    conda convert "$PACKAGE" --platform linux-64 -o packages
    cd packages
    LABEL=${2:-main}

    echo "Uploading source platform to anaconda with anaconda upload..."
    set +x # hide token
    anaconda -t "$CONDA_UPLOAD_TOKEN" upload -u "$1" --force "$PACKAGE" -l "$LABEL"
    set -x
    for os in $(ls); do
        cd $os
        for package in $(ls); do
            echo "Uploading $package to anaconda with anaconda upload..."
            set +x # hide token
            anaconda -t "$CONDA_UPLOAD_TOKEN" upload -u "$1" --force "$package" -l "$LABEL"
            set -x
        done
        cd ..
    done
    cd ..
    echo "Upload command complete!"
else
    echo "Upload cancelled due to failed test."
fi
