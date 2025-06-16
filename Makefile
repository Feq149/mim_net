.PHONY: train test sync copy
train:
	export PYTHONPATH=. && python3.11 scripts/train_mimnet.py
test:
	export PYTHONPATH=. && python3.11 scripts/evaluate_mimnet.py
plot:
	export PYTHONPATH=. && python3.11 scripts/coords.py
sync:
	mkdir -p checkpoints logs
	scp miracle:~/mim_net/checkpoints/*.pth checkpoints/ && scp -r miracle:~/mim_net/logs/* logs/
copy:
	rsync -av --no-group --exclude='.venv' --exclude='.*' --include='*/' --include='*.py' --exclude='*' ./ miracle:~/mim_net/
