cd $PYTHONPATH
# Step 3: Version the data sample using DVC
echo "Versioning the data sample..."
version=$(cat ../configs/data_version.yaml | shyaml get-value sample_version)
echo "Data sample version: v$version"
dvc add ../data/samples/sample.csv
git add ../data/samples/sample.csv.dvc
git commit -m "Add and version data sample"
git push origin main
git tag -a "$version" -m "add data version v$version"
git push --tags
echo "Data sample versioned"

echo "Process completed successfully"