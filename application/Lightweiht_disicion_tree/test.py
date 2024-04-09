from phe import paillier  # 开源库
import time  # 做性能测试
import  sys
# 测试paillier参数
print("默认私钥大小：", paillier.DEFAULT_KEYSIZE)  # 2048
# 生成公私钥
public_key, private_key = paillier.generate_paillier_keypair()
# 测试需要加密的数据
message_list = [3.1415926, 100, -4.6e-12,1,21,1,1,2,1,1,1,1,1,1,1,1]
print(sys.getsizeof(message_list))
# 加密操作
time_start_enc = time.time()
encrypted_message_list = [public_key.encrypt(m) for m in message_list]
print(sys.getsizeof(encrypted_message_list))
time_end_enc = time.time()
print("加密耗时s：", time_end_enc - time_start_enc)
# 解密操作
time_start_dec = time.time()
decrypted_message_list = [private_key.decrypt(c) for c in encrypted_message_list]
time_end_dec = time.time()
print("解密耗时s：", time_end_dec - time_start_dec)
# print(encrypted_message_list[0])
print("原始数据:", decrypted_message_list)

# Assuming the rest of your code remains the same up to this point

# # Select only the first three items for homomorphic operation tests
# a, b, c = encrypted_message_list[:3]  # This now correctly unpacks the first three items
#
# # Homomorphic operations
# a_sum = a + 5  # Encrypted message plus plaintext
# a_sub = a - 3  # Encrypted message plus the negation of plaintext
# b_mul = b * 1  # Encrypted message times plaintext
# c_div = c / -10.0  # Encrypted message times the reciprocal of plaintext
#
# # Continue with the decryption and printing of results as you have in your original code
#
#
# ##密文加密文
# print((private_key.decrypt(a) + private_key.decrypt(b)) == private_key.decrypt(a + b))
# # 报错，不支持a*b，因为通过密文加实现了明文加的目的，这和原理设计是不一致的，只支持密文加！
# print((private_key.decrypt(a) + private_key.decrypt(b)) == private_key.decrypt(a * b))