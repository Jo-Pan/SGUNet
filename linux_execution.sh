#python3 main.py --connection cc --exp multi-decode

# # baseline UNet cv
# python3 main.py --exp 0 --cv 0
# python3 main.py --exp 0 --cv 0 --mode test

# # baseline SGUNet cv
# python3 main.py --cv 0

# # 2020/03/05
# python3 main.py --cv 10

# # UNetPlusPlus
# python3 main.py --exp 0 --exp2 UNetPlusPlus --cv 0

# #SGUNet 22b
# python3 main.py --exp residual_decoder22b --cv 0

# #SGUNet 22b & multi loss
# python3 main.py --exp residual_decoder22b --loss multi-outputcedice --cv 0


# # baseline UNet cv
# python3 main.py --exp 0 --cv 0 --mode test --no_test_augmentation --GPU 1
# python3 main.py --exp 0 --cv 1 --mode test --no_test_augmentation --GPU 1
# python3 main.py --exp 0 --cv 2 --mode test --no_test_augmentation --GPU 1
# python3 main.py --exp 0 --cv 3 --mode test --no_test_augmentation --GPU 1
# python3 main.py --exp 0 --cv 4 --mode test --no_test_augmentation --GPU 1



# # # baseline SGUNet cv
# python3 main.py --cv 0 --mode test --no_test_augmentation --GPU 1
# python3 main.py --cv 1 --mode test --no_test_augmentation --GPU 1
# python3 main.py --cv 2 --mode test --no_test_augmentation --GPU 1
# python3 main.py --cv 3 --mode test --no_test_augmentation --GPU 1
# python3 main.py --cv 4 --mode test --no_test_augmentation --GPU 1


# # UNetPlusPlus
# python3 main.py --exp 0 --exp2 UNetPlusPlus --cv 0 --no_test_augmentation --GPU 1 --mode test
# python3 main.py --exp 0 --exp2 UNetPlusPlus --cv 1 --no_test_augmentation --GPU 1 --mode test
# python3 main.py --exp 0 --exp2 UNetPlusPlus --cv 2 --no_test_augmentation --GPU 1 --mode test
# python3 main.py --exp 0 --exp2 UNetPlusPlus --cv 3 --no_test_augmentation --GPU 1 --mode test
# python3 main.py --exp 0 --exp2 UNetPlusPlus --cv 4 --no_test_augmentation --GPU 1 --mode test

## attention UNet
python3 main.py --connection sa --no_test_augmentation --GPU 1 --cv 0 --exp 0