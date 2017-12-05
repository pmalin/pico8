
#dither_pattern = [
#[ 0,  1,  2,  3],
#[ 4,  5,  6,  7],
#[ 8,  9, 10, 11],
#[12, 13, 14, 15]  ]

dither_pattern = [
[ 11,  15,  1,  6 ],
[ 14,  0,  4,  8 ],
[ 5,  9, 12,  3 ],
[ 7, 10,  2,  13 ]  ]

dither_values = []

for i in range(0,15):
	dither_values.append(0)

for dy in range(0,len(dither_pattern)):
	for dx in range(0,len(dither_pattern[dy])):
		v = dither_pattern[dy][dx]
		for i in range(0,15):
			if (i>v):
				p = dx + dy * 4
				dither_values[i] = dither_values[i] + (1<<p)
				#print(str(i) + "+=" + str(1<<p))

result = "{"
first = True
for v in dither_values:
	if not first:
		result += ","
	result += str(v)
	first = False
result += "}"

print(result)