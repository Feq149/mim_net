# MimNet (zwijanie białek)

Projekt inspirowany grą FoldIt. Celem jest predykcja koordynatów 3D atomów C_alpha z użyciem sieci MimNet.

## Wymagania
- python 3.11
- `requirements.txt`
Wszystkie wymagane biblioteki można zainstalować poleceniem:

```bash
pip install -r requirements.txt
```
Pamiętaj, żeby wykonać
```
export PYTHONPATH=.
```


## Trening modelu

Aby rozpocząć trening modelu, użyj:
```
make train
```
lub bezpośrednio:
```
python scripts/train_mimnet.py
```

Model oraz logi TensorBoard zapisywane są w katalogach `checkpoints/` oraz `logs/`.
Możesz też skorzystać z gotowego modelu w `checkpoints/`.

## Ewaluacja

Aby przetestować wytrenowany model:
```
make test
```

## Wizualizacja

Do wizualizacji wyników możesz użyć:
```
make plot
```
W katalogu `images` będą pojawiać się obrazki.

## TensorBoard

Aby śledzić postęp treningu:
```
tensorboard --logdir logs
```
i otwórz podany adres w przeglądarce.

## Raport
Omówienie problemu znajduje się w pliku `raport.pdf`

---
Projekt został wykonany w ramach kursu Metody Probabilistyczne w Uczeniu Maszynowym na Uniwersytecie Jagiellońskim.