import struct
f = open('Behance_Image_Features.b', 'rb')
while True:
  itemId = f.read(8)
  feature = struct.unpack('f'*4096, f.read(4*4096))
  print(feature)
  print(len(feature))
  print(max(feature))