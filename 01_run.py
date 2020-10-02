#!/usr/bin/python3

import Config as cfg
import time, pathlib, numpy, torch
from torch.utils.data import DataLoader
from module import Embedding, Model, DataSet, DataCollate

def main():
    def randomseed():
        numpy.random.seed(123)
        torch.manual_seed(123)

    def get_fnames(dnames):
        F = {'trn'  : sorted(list(pathlib.Path(dnames['trn']).glob('*.npz'))),
             'test' : sorted(list(pathlib.Path(dnames['test']).glob('*.npz')))}
        return F

    def get_dataloader(fnames, N, bmask_data_mergin, device, batch_size, num_workers=1, shuffle=False, drop_last=False):
        dataloader = DataLoader(DataSet(fnames, N, bmask_data_mergin), pin_memory=True, num_workers=num_workers, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, collate_fn=DataCollate())
        return dataloader

    def init_model(dim, normfac, classnums, embeddims, N, max_state_dur, output_initial_bias, device):
        incode_embedding = Embedding(classnums, embeddims, device)
        model = Model(dim, normfac, incode_embedding, N, max_state_dur, output_initial_bias, device)
        return model

    def init_optimizer(model, hp):
        print('-- hyper parameters (optimizer) --')
        print('lr :', hp['lr'])
        return torch.optim.Adam(model.parameters(), hp['lr'])

    def training(ckptdir, F, model, optimizer, N, hp, device):
        def load_checkpoint(ckptdir, model, optimizer, siter):
            ckpt_iternums = sorted([int(str(p).split('_')[-1]) for p in pathlib.Path(ckptdir).glob('checkpoint_*')])
            if ckpt_iternums:
                print('-- checkpoint_' + str(ckpt_iternums[-1]) + ' load --')
                ckpt_path = ckptdir + '/checkpoint_' + str(ckpt_iternums[-1])
                ckpt_dict = torch.load(ckpt_path, map_location='cpu')
                model.load_state_dict(ckpt_dict['model'], strict=False)
                optimizer.load_state_dict(ckpt_dict['optimizer'])
                siter = ckpt_dict['iter']
            epoch_state_iternums = sorted([int(str(p).split('_')[-1]) for p in pathlib.Path(ckptdir).glob('epoch_state_*')])
            if epoch_state_iternums:
                epoch_state_path = ckptdir + '/epoch_state_' + str(epoch_state_iternums[-1])
                epoch_state_dict = torch.load(epoch_state_path, map_location='cpu')
                torch.random.set_rng_state(epoch_state_dict['torch_rng_state'])
            return model, optimizer, siter

        def run(ckptdir, B, model, optimizer, siter, trigger):
            pathlib.Path(ckptdir).mkdir(exist_ok=True, parents=True)
            i, ave_loss, start_time = siter, 0.0, time.perf_counter()
            while True:
                for X, A, T, M in B:
                    X, A, T, M = [x.to(device) for x in X], [a.to(device) for a in A], [t.to(device) for t in T], [(m0.to(device) if m0 is not None else None, m1.to(device) if m1 is not None else None) for m0, m1 in M]
                    optimizer.zero_grad()
                    loss = model.loss(X, A, T, M)
                    loss.backward()
                    optimizer.step()

                    i += 1

                    ave_loss += loss.item()
                    if 0 == i % trigger['logger']:
                        ave_loss = ave_loss / trigger['logger']
                        print('Iteration: {} loss: {:.6f} {:.3f}sec'.format(i, ave_loss, time.perf_counter() - start_time), flush=True)
                        ave_loss, start_time = 0.0, time.perf_counter()

                    if 0 == i % trigger['check']:
                        fname = ckptdir + '/checkpoint_' + str(i)
                        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'iter': i}, fname)

                    if i >= trigger['maxiter']:
                        break
                else:
                    if (len(B) >= trigger['check']) or ((i % trigger['check']) <= len(B)):
                        fname = ckptdir + '/epoch_state_' + str(i)
                        torch.save({'torch_rng_state': torch.random.get_rng_state()}, fname)
                    continue
                break
            fname = ckptdir + '/checkpoint_' + str(i)
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'iter': i}, fname)
            pathlib.Path(ckptdir+'/training_done').touch()

        def clean_checkpoint(ckptdir):
            print('-- clean --')
            ckpt_iternums = sorted([int(str(p).split('_')[-1]) for p in pathlib.Path(ckptdir).glob('checkpoint_*')])
            for n in ckpt_iternums[:-1]:
                pathlib.Path(ckptdir+'/checkpoint_'+str(n)).unlink()

        model, optimizer, siter = load_checkpoint(ckptdir, model, optimizer, siter=0)
        if not pathlib.Path(ckptdir+'/training_done').exists():
            B = get_dataloader(F['trn'], N, device=device, shuffle=True, drop_last=True, **hp['data'])
            run(ckptdir, B, model, optimizer, siter, **hp['trn'])
        if hp['clean']:
            clean_checkpoint(ckptdir)
        return model, optimizer

    def synthesis(gendir, F, model, N, ofeatdims, device):
        def _write_feat(gendir, s, y, ofeatdims):
            for ext, dim in ofeatdims.items():
                feat = y[:, dim]
                feat.tofile(gendir + '/' + s + '.' + ext)

        pathlib.Path(gendir).mkdir(exist_ok=True, parents=True)
        batch_size, num_workers = 1, 1
        B = get_dataloader(F, N, None, device, batch_size, num_workers, shuffle=False, drop_last=False)
        for i, (X, A, _, _) in enumerate(B):
            X, A = [x.to(device) for x in X], [a.to(device) for a in A]
            S = [pathlib.PurePath(f).stem for f in F[i*batch_size:(i+1)*batch_size]]
            for s, y in zip(S, model.output(X, A)):
                _write_feat(gendir, s, y.detach().cpu().numpy(), ofeatdims)

    device = torch.device(cfg.device)
    F = get_fnames(cfg.datadir)
    randomseed()

    print('--- Model initialization  ---')
    model = init_model(cfg.dim, cfg.normfac, cfg.classnums, cfg.embeddims, cfg.oN, cfg.max_state_dur, cfg.output_initial_bias, device)

    for k in cfg.hp.keys():
        print('--- Model training (' + k + ') ---')
        model.set_hyperparameter(cfg.hp[k]['model'], device)
        optimizer = init_optimizer(model, cfg.hp[k]['optimizer'])
        model, _ = training(cfg.ckptdir+'_'+k, F, model, optimizer, cfg.iN, cfg.hp[k], cfg.device)

    print('--- Synthesis ---')
    synthesis(cfg.gendir, F['test'], model, cfg.iN, cfg.ofeatdims, device)

    print('Done')

if __name__ == '__main__':
    main()
