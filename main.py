import time

import params
from utils import *

from fscil import FSCIL_IDS

from data.CICIDS2017.pktseq.process import Dataset as CICIDS2017
from data.CSECICIDS2018.pktseq.process import Dataset as CSECICIDS2018
from model import DynamicModel
        

def main():
    args = params.get_args()
    params.set_rootpth(args)
    params.set_outpth(args)
    setseeds(args.seed)
    args.gpu_num = setgpu(args.gpuseq)
    print(vars(args))

    res_list = [args]   
    rep_res_log = []
    
    tol_start_time = time.time()

    if args.dataset == 'cicids2017':
            dataset = CICIDS2017(
                root=args.root, 
                N=args.Np, 
                S=args.S
                )
    elif args.dataset == 'csecicids2018':
        dataset = CSECICIDS2018(
                root=args.root, 
                N=args.Np, 
                S=args.S
                )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    for i in range(args.repeat_n):
        print(f"-------------EXPERIMENT {i}-------------")
        res_list.append(f"-------------EXPERIMENT {i}-------------")
        exp_start_time = time.time()

        dataset.exp_init(args.base_cls,
                        rand_tr_smpl=args.rand_tr_tst_split,
                        feature_minmax=args.feature_minmax,
                        feature_norm=args.feature_norm
                    )

        model = DynamicModel(args, args.backbone, dataset)

        trainer = FSCIL_IDS(args, dataset, model)
        
        res_list_, res_dict = trainer.run()

        res_list.extend(res_list_)
        rep_res_log.append(res_dict)
        exp_time = time.time()-exp_start_time
        print(f"Experiment {i} finished in {exp_time:.2f}s")
        res_list.append(f"Experiment {i} finished in {exp_time:.2f}s")
    tol_time = time.time()-tol_start_time
    print(f"-------------ALL EXPERIMENTS FINISHED-------------")
    res_list.append(f"-------------ALL EXPERIMENTS FINISHED-------------")
    print(f"All experiments finished in {tol_time:.2f}s")
    res_list.append(f"All experiments finished in {tol_time:.2f}s")

    avglog = cal_log_avg(args, rep_res_log)

    print("-------------AVERAGE METRICS-------------")
    res_list.append("-------------AVERAGE METRICS-------------")

    res_list.append("Final Confusion Matrix:")
    res_list.append(np.array2string(avglog['conf_mat'][-1], separator=', '))

    res_list.append(f"Total\nAcc: {avglog['acc']}")
    res_list.append(f"Class-wise Avg Acc: {avglog['avg_cls_acc']}")
    res_list.append(f"Precision: {avglog['precision']}")
    res_list.append(f"Recall: {avglog['recall']}")
    res_list.append(f"F1: {avglog['F1']}")

    res_list.append(f"Base\nAcc: {avglog['base_acc']}")
    res_list.append(f"Class-wise Avg Acc: {avglog['base_avg_cls_acc']}")

    res_list.append(f"New\nAcc: {avglog['new_acc']}")
    res_list.append(f"Class-wise Avg Acc: {avglog['new_avg_cls_acc']}")

    res_list.append(f"Avg metrics of all sessions: \nacc: {sum(avglog['acc'])/len(avglog['acc'])}, prec: {sum(avglog['precision'])/len(avglog['precision'])}, recall: {sum(avglog['recall'])/len(avglog['recall'])}, F1: {sum(avglog['F1'])/len(avglog['F1'])}")
    res_list.append(f"PD: {avglog['acc'][0]-avglog['acc'][-1]}")

    res_list.append(f"Detection speed: {avglog['speed']} flows/s")

    print("Final Confusion Matrix:")
    print(avglog['conf_mat'][-1])

    print(f"Total\nAcc: {avglog['acc']}")
    print(f"Class-wise Avg Acc: {avglog['avg_cls_acc']}")
    print(f"Precision: {avglog['precision']}")
    print(f"Recall: {avglog['recall']}")
    print(f"F1: {avglog['F1']}")

    print(f"Base\nAcc: {avglog['base_acc']}")
    print(f"Class-wise Avg Acc: {avglog['base_avg_cls_acc']}")

    print(f"New\nAcc: {avglog['new_acc']}")
    print(f"Class-wise Avg Acc: {avglog['new_avg_cls_acc']}")

    print(f"Avg metrics of all sessions: \nacc: {sum(avglog['acc'])/len(avglog['acc'])}, prec: {sum(avglog['precision'])/len(avglog['precision'])}, recall: {sum(avglog['recall'])/len(avglog['recall'])}, F1: {sum(avglog['F1'])/len(avglog['F1'])}")
    print(f"PD: {avglog['acc'][0]-avglog['acc'][-1]}")
    
    print(f"Detection speed: {avglog['speed']} flows/s")

    tol_params = sum(p.numel() for p in trainer.model.parameters())
    trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print(f"model tol params: {tol_params}, trainable params: {trainable_params}")
    res_list.append(f"model tol params: {tol_params}, trainable params: {trainable_params}")

    save_cm_img(avglog['conf_mat'][-1], args.o_pth, dataset.label_map)
    save_log(os.path.join(args.o_pth, 'log.txt'), res_list)

if __name__ == '__main__':
    main()
