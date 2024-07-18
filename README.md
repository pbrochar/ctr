# ctr

## Usage

```python
@Contract(pre=["a == 1"],check_type=False)
def test(a: int, b: str):
    """
    ctr.pre:
        b != "test"
        len(b) > 3
    ctr.post:
         __return__ == None
    """
    print(a, b)
```