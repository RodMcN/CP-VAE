import argparse
from CPVAE import *
from dataset import *
import torch
import time
from glob import glob
import json
import os

if __name__ == "__main__":

    # use beta = 100, cap = 2, annleaing = 4

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dims", type=int, required=False, default=64)
    parser.add_argument("--densenet", action="store_true",)
    parser.add_argument("-e", "--epochs", type=int, required=False, default=50)
    parser.add_argument("-bs", "--batchsize", type=int, required=False, default=64)
    parser.add_argument("-opt", "--opt", type=str, required=False, default="adam")
    parser.add_argument("--beta", type=int, required=False, default=1)
    parser.add_argument("--annealing", type=int, required=False, default=1)
    parser.add_argument("--cap", type=int, required=False, default=1)
    parser.add_argument("--schedule", action="store_true",)
    parser.add_argument("--weightsfolder", type=str,)
    parser.add_argument("--weights", type=str, )
    parser.add_argument("-o", "--overwrite", action="store_true",)
    parser.add_argument("--cuda", action="store_true", default=True)
    parser.add_argument("--workers", type=int, required=False, default=2)

    args = parser.parse_args()
    lr = 1e-5
    device = "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
    assert args.dims in [64, 144, 256]
    opt = args.opt.lower()
    assert opt in ["adam", "adamw", "sgd", "rmsprop"]

    if args.densenet:
        block = DenseBlock
    else:
        block = ResBlock


    model = VAE(Encoder(latent_dims=args.dims), Decoder(latent_dims=args.dims, num_featues=672))
    if args.weights:
        model.load_state_dict(torch.load(args.weights))
    model.to(device)

    if opt == "adam":
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    elif opt == "adamw":
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    elif opt == "sgd":
        opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=lr)
    elif opt == "rmsprop":
        opt = torch.optim.RMSprop([p for p in model.parameters() if p.requires_grad], lr=lr)

    scheduler = None

    losses = {}

    scale = args.beta >= 1
    annealing = args.annealing
    kl_cap = args.cap
    scale_kl = args.beta

    imgs = glob('data/cp/split_and_merged/*.hdf5')
    meas = glob('data/cp/split_and_merged/meas/*.hdf5')
    imgs = sorted(imgs)
    meas = sorted(meas)
    try:
        train_id = np.loadtxt("data/train_split.txt").astype(np.int)
        test_id = np.loadtxt("data/test_split.txt").astype(np.int)
    except:
        if input("could not find train/test splits. Generate new splits? (y/n)\n").strip().lower() == 'y':

            train_id = np.arange(len(imgs))
            np.random.shuffle(train_id)

            test_id = train_id[1000:]
            train_id = train_id[:1000]

            np.savetxt("train_split_256.txt", train_id)
            np.savetxt("test_split_256.txt", test_id)
        else:
            raise
    train_imgs = np.array(imgs)[train_id]
    train_meas = np.array(meas)[train_id]
    test_imgs = np.array(imgs)[test_id]
    test_meas = np.array(meas)[test_id]
    train_dataset = Dataset(train_imgs, train_meas)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchsize, drop_last=True, num_workers=args.workers,
                                                    worker_init_fn=worker_init_fn)
    test_dataset = Dataset(test_imgs, test_meas)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batchsize, num_workers=args.workers, drop_last=True,
                                                   worker_init_fn=worker_init_fn)


    for e in range(args.epochs):
        if e == 1:
            opt.zero_grad()
            opt = torch.optim.Adam(model.parameters(), lr=lr*0.1)
            losses = {}
            if args.schedule:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10, eta_min=0)

        print(f"{e + 1}/{args.epochs}")

        if scale:
            scale = (e + 1) / annealing
            if scale > kl_cap:
                scale = kl_cap
            scale *= scale_kl

        k_losses = []
        r_losses = []
        m_losses = []
        total_losses = []

        t = time.time()

        for i, (x_img, x_meas) in enumerate(train_data_loader):
            x_img = x_img.to(device)
            x_meas = x_meas.to(device)

            opt.zero_grad()

            y_img, y_meas = model(x_img)

            loss, k_loss, r_loss = model.compute_loss(x_img, y_img, scale_kl=scale)
            f_loss = torch.nn.functional.l1_loss(y_meas, x_meas, reduction='sum') / args.batchsize
            loss = loss + f_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1000 / ((e * 2) + 1), norm_type=2)
            opt.step()

            k_losses.append(-k_loss)
            r_losses.append(-r_loss)
            m_losses.append(f_loss.item())
            total_losses.append(loss.item())

            t2 = time.time() - t
            m = int(t2 / 60)
            s = int(t2 % 60)

            print(
                f"\rtrain: l: {np.mean(total_losses):.3f}, k: {np.mean(k_losses):.3f}, r: {np.mean(r_losses):.3f}, m: {np.mean(m_losses):.3f}, t={m}m{s}s",
                end="\t", flush=True)

        print()
        losses[e] = {"train":
                         {"total": np.mean(total_losses), "k": np.mean(k_losses), "r": np.mean(r_losses),
                          "m": np.mean(m_losses), }
                     }

        with torch.no_grad():

            k_losses = []
            r_losses = []
            m_losses = []
            total_losses = []

            t = time.time()

            for i, (x_img, x_meas) in enumerate(test_data_loader):
                x_img = x_img.to(device)
                x_meas = x_meas.to(device)

                y_img, y_meas = model(x_img)

                loss, k_loss, r_loss = model.compute_loss(x_img, y_img, scale_kl=scale)
                f_loss = torch.nn.functional.l1_loss(y_meas, x_meas, reduction='sum') / args.batchsize
                loss = loss + f_loss

                k_losses.append(-k_loss)
                r_losses.append(-r_loss)
                m_losses.append(f_loss.item())
                total_losses.append(loss.item())

                t2 = time.time() - t
                m = int(t2 / 60)
                s = int(t2 % 60)

                print(
                    f"\rtest: l: {np.mean(total_losses):.3f}, k: {np.mean(k_losses):.3f}, r: {np.mean(r_losses):.3f}, m: {np.mean(m_losses):.3f}, t={m}m{s}s",
                    end="\t", flush=True)

            print("\n")
            if scheduler:
                scheduler.step()
                print("new lr:", opt.param_groups[0]['lr'])

            losses[e]["test"] = {"total": np.mean(total_losses), "k": np.mean(k_losses), "r": np.mean(r_losses),
                                 "m": np.mean(m_losses), }


        if args.overwrite:
            fname = "model.pt"
        else:
            fname = f"model_{e}.pt"
        if args.weightsfolder:
            path = args.weightsfolder
        else:
            path = "./models"

        with open(os.path.join(path, "losses.json"), "w") as f:
            json.dump(losses, f)
        torch.save(model.state_dict(), os.path.join(path, fname))


