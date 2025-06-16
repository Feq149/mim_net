.PHONY: train env test sync copy
env:
	. .venv/bin/activate && python3.11 -m pip install -r requirements.txt && export PYTHONPATH=.
train:
	python3.11 scripts/train_mimnet.py
test:
	python3.11 scripts/evaluate_mimnet.py
plot:
	python3.11 scripts/coords.py
sync:
	mkdir -p checkpoints logs
	scp miracle:~/mim_net/checkpoints/*.pth checkpoints/ && scp -r miracle:~/mim_net/logs/* logs/
copy:
	rsync -av --no-group --exclude='.venv' --exclude='.*' --include='*/' --include='*.py' --exclude='*' ./ miracle:~/mim_net/
