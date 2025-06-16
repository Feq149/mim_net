.PHONY: train test sync copy
train:
	set PYTHONPATH=. && python scripts/train_mimnet.py
test:
	set PYTHONPATH=. && python scripts/evaluate_mimnet.py
plot:
	set PYTHONPATH=. && python3.11 scripts/coords.py
sync:
	mkdir -p checkpoints logs
	scp miracle:~/mim_net/checkpoints/*.pth checkpoints/ && scp -r miracle:~/mim_net/logs/* logs/
copy:
	rsync -av --no-group --exclude='.venv' --exclude='.*' --include='*/' --include='*.py' --exclude='*' ./ miracle:~/mim_net/
