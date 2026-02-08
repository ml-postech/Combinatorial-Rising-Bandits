You can run the following commands in order to set up the environment:

```
conda create -n myenv python=3.9
conda activate myenv
pip install -r requirements.txt
```

You can run the script for experiments, e.g., the 'Toy' task:

```
./scripts/Toy.sh
```

If it doesn't run, it might be a permission issue. Please run:

```
chmod +777 ./scripts/Toy.sh
```