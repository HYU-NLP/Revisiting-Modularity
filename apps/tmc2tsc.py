from eval.apps_metric import apps_metric
from datasets import Dataset

tsc_27 = [
    [
        """import math

for _ in range(int(input())):
    player1_distance, player2_distance = map(int, input().split())
    
    jumps = 0
    distance = player1_distance
    while distance >= 0:
        if distance == 0:
            jumps += 1
            break
        power = int(math.log2(distance + 1))
        if power == 0:
            jumps += 1
            break
        step = (2 ** power) - 1
        distance -= step + 1
        if distance == -1:
            jumps += 1
            break
        jumps += 1
    player1_jumps = jumps
    
    jumps = 0
    distance = player2_distance
    while distance >= 0:
        if distance == 0:
            jumps += 1
            break
        power = int(math.log2(distance + 1))
        if power == 0:
            jumps += 1
            break
        step = (2 ** power) - 1
        distance -= step + 1
        if distance == -1:
            jumps += 1
            break
        jumps += 1
    player2_jumps = jumps
    
    if player1_jumps == player2_jumps:
        winner = 0
        difference = 0
    elif player1_jumps < player2_jumps:
        winner = 1
        difference = player2_jumps - player1_jumps
    else:
        winner = 2
        difference = player1_jumps - player2_jumps
    
    print(winner, difference)"""
    ],
    [
        """import bisect
import sys

def hotel_distances(n, a, d):
    graph = [[0 for i in range(n+1)] for j in range(18)]
    for i in range(n):
        x = bisect.bisect_right(a, a[i] + d)
        graph[0][i+1] = x
    for j in range(1, 18):
        for i in range(n):
            t = graph[j-1][i+1]
            graph[j][i+1] = graph[j-1][t]
    return graph

def calculate_days_to_travel(graph, x, y):
    x, y = min(x, y), max(x, y)
    days = 0
    for j in range(18)[::-1]:
        if graph[j][x] < y:
            days += 2**j
            x = graph[j][x]
        if j == 0 and x < y:
            days += 1
    return days

n = int(input())
a = list(map(int, input().split()))
d = int(input())
graph = [[0 for i in range(n+1)] for j in range(18)]
for i in range(n):
    x = bisect.bisect_right(a, a[i] + d)
    graph[0][i+1] = x
for j in range(1, 18):
    for i in range(n):
        t = graph[j-1][i+1]
        graph[j][i+1] = graph[j-1][t]

q = int(input())
for _ in range(q):
    x, y = map(int, input().split())
    x, y = min(x, y), max(x, y)
    days = 0
    for j in range(18)[::-1]:
        if graph[j][x] < y:
            days += 2**j
            x = graph[j][x]
        if j == 0 and x < y:
            days += 1
    print(days)"""
    ],
]

tsc_42 = [
    [
        """t = int(input())
test_cases = []
for _ in range(t):
    n = int(input())
    jars = list(map(int, input().split()))
    test_cases.append((n, jars))
result = []
for i in range(t):
    n = test_cases[i][0]
    ls = test_cases[i][1]
    oneneed = 2 * (n - ls.count(1))
    ldct = {0: 0}
    rdct = {0: 0}
    ctr = 0
    eaten = 0
    for j in range(n-1, -1, -1):
        eaten += 1
        ctr += (1 if ls[j] == 2 else -1)
        if ctr not in ldct:
            ldct[ctr] = eaten
    ctr = 0
    eaten = 0
    for j in range(n, 2*n):
        eaten += 1
        ctr += (1 if ls[j] == 2 else -1)
        if ctr not in rdct:
            rdct[ctr] = eaten
    best = 99**99
    for k in list(rdct.keys()):
        otk = oneneed - k
        if otk in ldct:
            best = min(best, rdct[k] + ldct[otk])
    result.append(best)
for res in result:
    print(res)"""
    ],
    [
        """n = int(input())
arr = list(map(int, input().split()))
q = int(input())

while q:
    q -= 1
    ar = input().split()
    t = ar[0]
    l = int(ar[1]) - 1
    r = int(ar[2])
    
    if t == 'U':
        arr[l] = r
    elif t == 'A':
        print(sum(arr[l:r]))
    elif t == 'M':
        print(max(arr[l:r]))
    elif t == 'm':
        print(min(arr[l:r]))
    elif t == 'S':
        max_val = max(arr[l:r])
        second_max = -1
        for i in range(l, r):
            if arr[i] < max_val and arr[i] > second_max:
                second_max = arr[i]
        print(second_max)
    elif t == 's':
        min_val = min(arr[l:r])
        second_min = 1000000000000
        for i in range(l, r):
            if arr[i] > min_val and arr[i] < second_min:
                second_min = arr[i]
        print(second_min)"""
    ],
]

tsc_101 = [
    [
        """n, m, k = list(map(int, input().split()))
if k == 1:
    x, y = 0, 0
    for p in range(2, n + 1):
        x += (n - p + 1)
    for p in range(2, m + 1):
        y += (m - p + 1)
    ans = x * y
    x = 0
    for p in range(1, n + 1):
        x += (n - p + 1)
    y = 0
    for p in range(1, m + 1):
        y += (m - p + 1)
    ans += m * x
    ans += n * y
    ans -= n * m
else:
    x, y = 0.0, 0.0
    q = 1.0
    for p in range(2, n + 1):
        q /= k * k
        x += (n - p + 1) * q
    for p in range(2, m + 1):
        q /= k * k
        y += (m - p + 1) * q
    ans = k * x * y
    x = 0.0
    q = 1.0
    for p in range(1, n + 1):
        x += (n - p + 1) * q
        q /= k
    y = 0.0
    q = 1.0
    for p in range(1, m + 1):
        y += (m - p + 1) * q
        q /= k
    ans += m * x
    ans += n * y
    ans -= n * m
    ans += 1e-9
print("%.0f" % ans)"""
    ],
    [
        """import math

q = int(input())
for _ in range(q):
    AB = [int(i) for i in input().split()]
    AB.sort()
    ab = AB[0] * AB[1]

    x = 0
    for i in range(int(math.sqrt(ab)), AB[1] + 1):
        if i * i >= ab:
            x = i - 1
            break
    if x == 0:
        result = 0
        continue

    for i in range(int(ab / x), ab + 1):
        if x * i >= ab:
            y = i - 1
            break
    
    cnt1 = 2 * x
    cnts = 1 if x == y else 0
    cntdd = 1 if x - AB[0] >= 0 else 0
    cntdu = 1 if AB[1] - y >= 0 and (AB[1] + 1) * (x - AB[1] + y) >= ab else 0
    result = cnt1 - cnts - cntdd - cntdu
    print(result)"""
    ],
]

tsc_134 = [
    [
        """import math

T = int(input())
while T:
    T -= 1
    N = int(input())
    xh = []
    m = []
    for _ in range(N):
        a, b = map(int, input().split())
        xh.append(a)
        m.append(b)
    
    res = []
    for even in [0, 1]:
        x = [x for x in xh]
        y = [y for y in m]
        up = []
        lw = []
        L = None
        R = None
        valid = True
        for i in range(0, len(x) - 1):
            if (i % 2 == even):
                if (y[i] - y[i + 1]) == 0:
                    if x[i] < x[i + 1]:
                        lw.append(0)
                    else:
                        valid = False
                        break
                else:
                    if y[i] < y[i + 1]:
                        l = (x[i + 1] - x[i]) / (y[i] - y[i + 1])
                        l = int(math.floor(l)) + 1
                        lw.append(max(0, l))
                    else:
                        r = (x[i + 1] - x[i]) / (y[i] - y[i + 1])
                        r = int(math.ceil(r)) - 1
                        if r < 0:
                            valid = False
                            break
                        up.append(r)
            else:
                if (y[i] - y[i + 1]) == 0:
                    if x[i] > x[i + 1]:
                        lw.append(0)
                    else:
                        valid = False
                        break
                else:
                    if y[i] > y[i + 1]:
                        l = (x[i + 1] - x[i]) / (y[i] - y[i + 1])
                        l = int(math.floor(l)) + 1
                        lw.append(max(0, l))
                    else:
                        r = (x[i + 1] - x[i]) / (y[i] - y[i + 1])
                        r = int(math.ceil(r)) - 1
                        if r < 0:
                            valid = False
                            break
                        up.append(r)
        if not valid:
            continue
        if len(lw) > 0:
            L = max(lw)
        else:
            L = 0
        if len(up) > 0:
            R = min(up)
            if L > R:
                continue
        else:
            R = float('inf')
        res.append((L, R))
    
    res.sort()
    sz = len(res)
    if N == 1:
        print("1")
        print("0 Inf")
    else:
        if sz == 2 and (res[0][1] + 1 == res[1][0]):
            print("1")
            print(res[0][0], res[1][1])
        else:
            print(sz)
            for interval in res:
                L, R = interval
                if R == float('inf'):
                    print(L, "Inf")
                else:
                    print(L, R)"""
    ],
    [
        """import bisect
import sys

n = int(input())
a = list(map(int, input().split()))
d = int(input())

graph = [[0 for _ in range(n + 1)] for _ in range(18)]
for i in range(n):
    x = bisect.bisect_right(a, a[i] + d)
    graph[0][i + 1] = x
for j in range(1, 18):
    for i in range(n):
        t = graph[j - 1][i + 1]
        graph[j][i + 1] = graph[j - 1][t]

q = int(input())
for _ in range(q):
    x, y = map(int, input().split())
    x, y = min(x, y), max(x, y)
    ans = 0
    
    for j in range(18)[::-1]:
        if graph[j][x] < y:
            ans += 2 ** j
            x = graph[j][x]
        if j == 0 and x < y:
            ans += 1
    print(ans)"""
    ],
]

tsc_169 = [
    [
        """from sys import stdin, stdout
from collections import defaultdict

for _ in range(int(stdin.readline())):
    n = int(stdin.readline())
    lst = list(map(int, stdin.readline().split()))

    n = len(lst)
    prefix_odd = [0] * n
    prefix_even = [0] * n
    odd_val = 0
    even_val = 0
    for i in range(n):
        if lst[i] % 2 == 0:
            even_val += 1
        else:
            odd_val += 1
        prefix_even[i] = even_val
        prefix_odd[i] = odd_val

    prefix_sum = [0] * len(lst)
    s = 0
    for i in range(len(lst)):
        s += lst[i]
        prefix_sum[i] = s

    element_index_map = {}
    count = {}
    for i in range(len(lst)):
        if lst[i] not in element_index_map:
            element_index_map[lst[i]] = i
            count[lst[i]] = 1
        else:
            element_index_map[lst[i]] = i
            count[lst[i]] += 1

    max_sum = 0
    graph = defaultdict(list)
    for i in range(len(lst)):
        graph[lst[i]].append(i)
    for key in graph:
        if len(graph[key]) > 1:
            prev = graph[key][0]
            for j in range(1, len(graph[key])):
                index2 = graph[key][j]
                index1 = prev
                prev = index2
                if key % 2 == 0:
                    val = prefix_even[index2] - prefix_even[index1] - 1
                    if val % 2 == 0:
                        temp_sum = prefix_sum[index2] - prefix_sum[index1] - key
                        if temp_sum > max_sum:
                            max_sum = temp_sum
                else:
                    val = prefix_odd[index2] - prefix_odd[index1] - 1
                    if val % 2 != 0:
                        temp_sum = prefix_sum[index2] - prefix_sum[index1] - key
                        if temp_sum > max_sum:
                            max_sum = temp_sum
    stdout.write(str(max_sum) + "\n")
"""
    ],
    [
        """import sys
input = sys.stdin.readline

inp = int(input())
for _ in range(inp):
    n, m = [int(w) for w in input().split()]
    matrix = []
    for __ in range(n):
        matrix.append([int(w) for w in input().split()])
    s = input().strip()
    p, q = [int(w) for w in input().split()]
    ans = 0
    for i in range(n + m - 1):
        dt = {0: 0, 1: 0}
        if i < m:
            row, col = 0, i
        else:
            row, col = i - m + 1, m - 1
        while col >= 0 and row < n:
            dt[matrix[row][col]] += 1
            col -= 1
            row += 1
        if s[i] == '0':
            t = min(dt[1] * p, q + dt[0] * p)
        elif s[i] == '1':
            t = min(dt[0] * p, q + dt[1] * p)
        ans += t
    print(ans)"""
    ],
]

variables = {
    "tsc_27": tsc_27,
    "tsc_42": tsc_42,
    "tsc_101": tsc_101,
    "tsc_134": tsc_134,
    "tsc_169": tsc_169,
}

eval_apps = apps_metric()

for seed in [27, 42, 101, 134, 169]:
    data = Dataset.from_json(
        f"data/2shot_demonstration_{seed}seed.json"
    )
    data = data.add_column("transformed_sc", variables[f"tsc_{seed}"])
    results, metrics = eval_apps._compute(
        data, [1], split="train", column_name="transformed_sc"
    )
    print(results)
    data.to_json(
        f"data/2shot_demonstration_{seed}seed.json"
    )
