from argparse import ArgumentParser

args = ArgumentParser()
args.add_argument('--train_data_path', type=str, default='dataset/big_matrix_train.npy')
args.add_argument('--test_data_path', type=str, default='dataset/big_matrix_test.npy')
args.add_argument('--captions_path', type=str, default='dataset/caption.csv')


args.add_argument('--max_caption_len', type=int, default=100)
args.add_argument('--max_category_len', type=int, default=10)
args.add_argument('--max_topic_len', type=int, default=50)
args.add_argument('--max_token_len', type=int, default=152)
args.add_argument('--model_name', type=str, default="models/chaoscodes/tinyllama-1___1b-step-50k-105b")

args.add_argument('--epoch_num', type=int, default=16)
args.add_argument('--batch_size_train', type=int, default=6)
args.add_argument('--batch_size_test', type=int, default=128)
args.add_argument('--device', type=str, default="cuda")
args.add_argument('--metric_list', type=list, default=[50,100,1000])
args.add_argument('--num_eval', type=int, default=2)

args.add_argument('--temperature', type=float, default= 0.07)
args.add_argument('--learning_rate', type=float, default=1e-4)
args.add_argument('--weight_decay', type=float, default=0)
args.add_argument('--warmup_steps', type=int, default=100)
args.add_argument('--gradient_accumulation_steps', type=int, default=100)

args.add_argument('--resume', type=bool, default=False)
args.add_argument('--checkpoint_path', type=str, default='checkpoint/9.pth')