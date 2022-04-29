
####################################################################################

# Avi ran this on April 28th on one 2080ti on CML and it took 6.5 hrs

#python train.py --dataset RAVEN-FAIR --path /cmlscratch/avi1/RAVENFAIR/RAVENFAIR/ --wd 1e-5 --num_workers 4 --levels 111 --no_rc --epochs 20

# The final output here:
#Early stopping countdown: 0/20 (Best VAL: 88.54452, Best VAL TEST: 88.54452, Best TEST: 88.54452)
#Done Training
#Best Validation Accuracy: 88.54452446273434
#Best Validation Test Accuracy: 88.54452446273434
#Best Test Accuracy: 88.54452446273434
#Val In Regime:
#center_single: 94.640 /  94.640
#distribute_four: 77.100 /  77.100
#distribute_nine: 78.540 /  78.540
#in_center_single_out_center_single: 97.530 /  97.530
#in_distribute_four_out_center_single: 76.342 /  76.342
#left_center_single_right_center_single: 97.970 /  97.970
#up_center_single_down_center_single: 97.670 /  97.670

####################################################################################

python train.py --dataset RAVEN-FAIR --path /cmlscratch/avi1/RAVENFAIR/RAVENFAIR/ --wd 1e-5 --num_workers 4 --levels 111 --no_rc --epochs 20 --model_name resnet18 --testname resnet18

python train.py --dataset RAVEN-FAIR --path /cmlscratch/avi1/RAVENFAIR/RAVENFAIR/ --wd 1e-5 --num_workers 4 --levels 111 --no_rc --epochs 20 --model_name dt_net --testname dt_net

python train.py --dataset RAVEN-FAIR --path /cmlscratch/avi1/RAVENFAIR/RAVENFAIR/ --wd 1e-5 --num_workers 4 --levels 111 --no_rc --epochs 20 --model_name dt_net_recall --testname dt_net_recall
