import os
import torch
import numpy as np
import torch.nn.functional as F

def save_model(model, checkpoint, num, is_epoch=True):
    if not os.path.exists(checkpoint):
        os.system('mkdir -p '+ checkpoint)
    if is_epoch:
        torch.save(model.state_dict(), os.path.join(checkpoint, 'epoch_{:04d}.pth'.format(num)))
    else:
        torch.save(model.state_dict(), os.path.join(checkpoint, 'best.pth'.format(num)))

def load_model(model, resume):
    pretrained_dict = torch.load(resume)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


# def load_img(filepath):
#     image = imread(filepath, mode='RGB')
#     image = torch.from_numpy(
#         np.expand_dims(
#             np.transpose(image.astype(np.float32) / 255.0, (2, 0, 1)), 0))
#     return image

# def save_img(image, filepath):
#     imsave(filepath, np.squeeze(image.mul(255).round().permute(0, 2, 3, 1).cpu().numpy().clip(0, 255)))


# def get_psnr_ssim(var_tar, var_pred):
#     return evaluate.get_psnr_ssim(
#         var_tar.data.mul(255).round().permute(0, 2, 3, 1).cpu().numpy().clip(0, 255),
#         var_pred.data.mul(255).round().permute(0, 2, 3, 1).cpu().numpy().clip(0, 255))

# def flowpred(framei, flow):
#     nh, nw = flow.shape[2:4]
#     grid = torch.from_numpy(np.stack(np.meshgrid(np.arange(0, nw), np.arange(0, nh), indexing='xy'))[np.newaxis, ...].astype(np.float32)).cuda()
#     flow = (flow + grid).permute(0, 2, 3, 1)
#     flow[..., 0] = flow[..., 0] / (nw - 1) * 2 - 1
#     flow[..., 1] = flow[..., 1] / (nh - 1) * 2 - 1
#     framej = F.grid_sample(framei, flow, mode='bilinear', padding_mode='zeros').round().clamp(0, 255)
#     return framej

# def arithmetic_compress(freqs, inp, bitout):
#     #inp:     CxN  or    N
#     #freqs: SxC    or  SXN
#     assert inp.shape[0] == freqs.shape[1]
#     enc = arithmeticcoding.ArithmeticEncoder(bitout)
#
#     if len(inp.shape) == 1:
#         # each elements has its own freqstable
#         # freqs0 = freqs
#         freqs = arithmeticcoding.MultiFrequencyTable(freqs)
#         for idx, symbol in enumerate(inp):
#             # freqs_i = arithmeticcoding.SimpleFrequencyTable(freqs0[:, idx])
#             enc.write(freqs, symbol)
#             if idx == inp.shape[0]:
#                 enc.write(freqs, 0) # EOF
#             freqs.shift_no_eof(1)
#
#     elif len(inp.shape) == 2:
#         # all elements within a channel have the same freqstable
#         freqs = arithmeticcoding.MultiFrequencyTable(freqs)
#         for c in range(inp.shape[0]):
#             for i in range(inp.shape[1]):
#                 symbol = inp[c, i]
#                 enc.write(freqs, symbol)
#             if c == inp.shape[0] - 1:
#                 enc.write(freqs, 0)  # EOF
#             freqs.shift_no_eof(1)
#
#     enc.finish()
#     bitout.close()
#     return bitout.output[0: bitout.index+4]
#
#
# def arithmetic_decompress(freqs, bitin, outp):
#     # outp:    CxN  or    N
#     # freqs: SxC    or  SXN
#     assert outp.shape[0] == freqs.shape[1]
#     dec = arithmeticcoding.ArithmeticDecoder(bitin)
#
#     if len(outp.shape) == 1:
#         # all elements have the same freqstable
#         freqs = arithmeticcoding.MultiFrequencyTable(freqs)
#         for i in range(outp.shape[0]):
#             symbol = dec.read(freqs)
#             if symbol == 0:  # EOF
#                 break
#             outp[i] = symbol
#             freqs.shift_no_eof(1)
#
#     elif len(outp.shape) == 2:
#         # all elements within a channel have the same freqstable
#         freqs = arithmeticcoding.MultiFrequencyTable(freqs)
#         for c in range(outp.shape[0]):
#             for i in range(outp.shape[1]):
#                 symbol = dec.read(freqs)
#                 if symbol == 0:  # EOF
#                     break
#                 outp[c, i] = symbol
#             freqs.shift_no_eof(1)
#
#     return outp