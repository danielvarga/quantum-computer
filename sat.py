import itertools

# taken from here: https://cstheory.stackexchange.com/questions/38538/oracle-construction-for-grovers-algorithm/38551#38551
# supposedly implemented here:
# https://algassert.com/quirk#circuit=%7B%22cols%22%3A%5B%5B%22X%22%2C%22X%22%2C%22X%22%2C%22X%22%5D%2C%5B%22H%22%2C%22H%22%2C%22H%22%2C%22H%22%5D%2C%5B%22%E2%80%A2%22%2C1%2C%22%E2%80%A2%22%2C%22%E2%80%A2%22%2C%22X%22%5D%2C%5B1%2C%22%E2%97%A6%22%2C%22%E2%97%A6%22%2C%22%E2%80%A2%22%2C1%2C%22X%22%5D%2C%5B%22%E2%97%A6%22%2C%22%E2%80%A2%22%2C1%2C%22%E2%97%A6%22%2C1%2C1%2C%22X%22%5D%2C%5B%22%E2%80%A2%22%2C%22%E2%97%A6%22%2C%22%E2%80%A2%22%2C1%2C1%2C1%2C1%2C%22X%22%5D%2C%5B1%2C1%2C1%2C1%2C%22%E2%97%A6%22%2C%22%E2%97%A6%22%2C%22%E2%97%A6%22%2C%22%E2%97%A6%22%2C%22X%22%5D%2C%5B1%2C1%2C1%2C1%2C1%2C1%2C1%2C1%2C%22Z%22%5D%2C%5B1%2C1%2C1%2C1%2C1%2C1%2C1%2C1%2C%22Chance%22%5D%2C%5B1%2C1%2C1%2C1%2C%22%E2%97%A6%22%2C%22%E2%97%A6%22%2C%22%E2%97%A6%22%2C%22%E2%97%A6%22%2C%22X%22%5D%2C%5B%22%E2%80%A2%22%2C%22%E2%97%A6%22%2C%22%E2%80%A2%22%2C1%2C1%2C1%2C1%2C%22X%22%5D%2C%5B%22%E2%97%A6%22%2C%22%E2%80%A2%22%2C1%2C%22%E2%97%A6%22%2C1%2C1%2C%22X%22%5D%2C%5B1%2C%22%E2%97%A6%22%2C%22%E2%97%A6%22%2C%22%E2%80%A2%22%2C1%2C%22X%22%5D%2C%5B%22%E2%80%A2%22%2C1%2C%22%E2%80%A2%22%2C%22%E2%80%A2%22%2C%22X%22%5D%2C%5B%22%E2%8A%96%22%2C%22%E2%8A%96%22%2C%22%E2%8A%96%22%2C%22X%22%5D%2C%5B%22%E2%80%A2%22%2C1%2C%22%E2%80%A2%22%2C%22%E2%80%A2%22%2C%22X%22%5D%2C%5B1%2C%22%E2%97%A6%22%2C%22%E2%97%A6%22%2C%22%E2%80%A2%22%2C1%2C%22X%22%5D%2C%5B%22%E2%97%A6%22%2C%22%E2%80%A2%22%2C1%2C%22%E2%97%A6%22%2C1%2C1%2C%22X%22%5D%2C%5B%22%E2%80%A2%22%2C%22%E2%97%A6%22%2C%22%E2%80%A2%22%2C1%2C1%2C1%2C1%2C%22X%22%5D%2C%5B1%2C1%2C1%2C1%2C%22%E2%97%A6%22%2C%22%E2%97%A6%22%2C%22%E2%97%A6%22%2C%22%E2%97%A6%22%2C%22X%22%5D%2C%5B1%2C1%2C1%2C1%2C1%2C1%2C1%2C1%2C%22Z%22%5D%2C%5B1%2C1%2C1%2C1%2C1%2C1%2C1%2C1%2C%22Chance%22%5D%2C%5B1%2C1%2C1%2C1%2C%22%E2%97%A6%22%2C%22%E2%97%A6%22%2C%22%E2%97%A6%22%2C%22%E2%97%A6%22%2C%22X%22%5D%2C%5B%22%E2%80%A2%22%2C%22%E2%97%A6%22%2C%22%E2%80%A2%22%2C1%2C1%2C1%2C1%2C%22X%22%5D%2C%5B%22%E2%97%A6%22%2C%22%E2%80%A2%22%2C1%2C%22%E2%97%A6%22%2C1%2C1%2C%22X%22%5D%2C%5B1%2C%22%E2%97%A6%22%2C%22%E2%97%A6%22%2C%22%E2%80%A2%22%2C1%2C%22X%22%5D%2C%5B%22%E2%80%A2%22%2C1%2C%22%E2%80%A2%22%2C%22%E2%80%A2%22%2C%22X%22%5D%2C%5B%22%E2%8A%96%22%2C%22%E2%8A%96%22%2C%22%E2%8A%96%22%2C%22X%22%5D%2C%5B%22%E2%80%A2%22%2C1%2C%22%E2%80%A2%22%2C%22%E2%80%A2%22%2C%22X%22%5D%2C%5B1%2C%22%E2%97%A6%22%2C%22%E2%97%A6%22%2C%22%E2%80%A2%22%2C1%2C%22X%22%5D%2C%5B%22%E2%97%A6%22%2C%22%E2%80%A2%22%2C1%2C%22%E2%97%A6%22%2C1%2C1%2C%22X%22%5D%2C%5B%22%E2%80%A2%22%2C%22%E2%97%A6%22%2C%22%E2%80%A2%22%2C1%2C1%2C1%2C1%2C%22X%22%5D%2C%5B1%2C1%2C1%2C1%2C%22%E2%97%A6%22%2C%22%E2%97%A6%22%2C%22%E2%97%A6%22%2C%22%E2%97%A6%22%2C%22X%22%5D%2C%5B1%2C1%2C1%2C1%2C1%2C1%2C1%2C1%2C%22Z%22%5D%2C%5B1%2C1%2C1%2C1%2C1%2C1%2C1%2C1%2C%22Chance%22%5D%5D%7D
formula_original = [
'x xx',
' oox',
'ox o',
'xox ',
]

# supposedly implemented here:
# https://algassert.com/quirk#circuit={%22cols%22:[[%22X%22,%22X%22,%22X%22,%22X%22],[%22H%22,%22H%22,%22H%22,%22H%22],[%22%E2%80%A2%22,1,1,%22%E2%80%A2%22,%22X%22],[1,1,%22%E2%97%A6%22,1,1,%22X%22],[%22%E2%97%A6%22,%22%E2%80%A2%22,1,%22%E2%97%A6%22,1,1,%22X%22],[1,%22%E2%97%A6%22,%22%E2%80%A2%22,1,1,1,1,%22X%22],[1,1,1,1,%22%E2%97%A6%22,%22%E2%97%A6%22,%22%E2%97%A6%22,%22%E2%97%A6%22,%22X%22],[1,1,1,1,1,1,1,1,%22Z%22],[1,1,1,1,1,1,1,1,%22Chance%22],[1,1,1,1,%22%E2%97%A6%22,%22%E2%97%A6%22,%22%E2%97%A6%22,%22%E2%97%A6%22,%22X%22],[1,%22%E2%97%A6%22,%22%E2%80%A2%22,1,1,1,1,%22X%22],[%22%E2%97%A6%22,%22%E2%80%A2%22,1,%22%E2%97%A6%22,1,1,%22X%22],[1,1,%22%E2%97%A6%22,1,1,%22X%22],[%22%E2%80%A2%22,1,1,%22%E2%80%A2%22,%22X%22],[%22%E2%8A%96%22,%22%E2%8A%96%22,%22%E2%8A%96%22,%22X%22],[%22%E2%80%A2%22,1,1,%22%E2%80%A2%22,%22X%22],[1,1,%22%E2%97%A6%22,1,1,%22X%22],[%22%E2%97%A6%22,%22%E2%80%A2%22,1,%22%E2%97%A6%22,1,1,%22X%22],[1,%22%E2%97%A6%22,%22%E2%80%A2%22,1,1,1,1,%22X%22],[1,1,1,1,%22%E2%97%A6%22,%22%E2%97%A6%22,%22%E2%97%A6%22,%22%E2%97%A6%22,%22X%22],[1,1,1,1,1,1,1,1,%22Z%22],[1,1,1,1,1,1,1,1,%22Chance%22],[1,1,1,1,%22%E2%97%A6%22,%22%E2%97%A6%22,%22%E2%97%A6%22,%22%E2%97%A6%22,%22X%22],[1,%22%E2%97%A6%22,%22%E2%80%A2%22,1,1,1,1,%22X%22],[%22%E2%97%A6%22,%22%E2%80%A2%22,1,%22%E2%97%A6%22,1,1,%22X%22],[1,1,%22%E2%97%A6%22,1,1,%22X%22],[%22%E2%80%A2%22,1,1,%22%E2%80%A2%22,%22X%22],[%22%E2%8A%96%22,%22%E2%8A%96%22,%22%E2%8A%96%22,%22X%22],[%22%E2%80%A2%22,1,1,%22%E2%80%A2%22,%22X%22],[1,1,%22%E2%97%A6%22,1,1,%22X%22],[%22%E2%97%A6%22,%22%E2%80%A2%22,1,%22%E2%97%A6%22,1,1,%22X%22],[1,%22%E2%97%A6%22,%22%E2%80%A2%22,1,1,1,1,%22X%22],[1,1,1,1,%22%E2%97%A6%22,%22%E2%97%A6%22,%22%E2%97%A6%22,%22%E2%97%A6%22,%22X%22],[1,1,1,1,1,1,1,1,%22Z%22],[1,1,1,1,1,1,1,1,%22Chance%22]]}
formula = [
'x  x',
'  o ',
'ox o',
' ox ',
]

def sat_clause(clause, assignment):
    assert len(clause) == len(assignment)
    for l, a in zip(clause, assignment):
        if l=='o' and a:
            return True
        elif l=='x' and not a:
            return True
    return False

def sat(formula, assignment):
    for clause in formula:
        if not sat_clause(clause, assignment):
            return False
    return True

n = len(formula[0])
m = len(formula)

for assignment in itertools.product(*[[0, 1] for _ in range(n)]):
    if sat(formula, assignment):
        print(assignment)