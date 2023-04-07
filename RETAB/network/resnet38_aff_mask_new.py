import torch
import torch.nn as nn
import torch.sparse as sparse
import torch.nn.functional as F

import network.resnet38d
from tool import pyutils

class Net(network.resnet38d.Net):
    def __init__(self):
        super(Net, self).__init__()

        self.f8_3 = torch.nn.Conv2d(512, 64, 1, bias=False)
        self.f8_4 = torch.nn.Conv2d(1024, 128, 1, bias=False)
        self.f8_5 = torch.nn.Conv2d(4096, 256, 1, bias=False)

        self.f9 = torch.nn.Conv2d(448, 448, 1, bias=False)
        
        torch.nn.init.kaiming_normal_(self.f8_3.weight)
        torch.nn.init.kaiming_normal_(self.f8_4.weight)
        torch.nn.init.kaiming_normal_(self.f8_5.weight)
        torch.nn.init.xavier_uniform_(self.f9.weight, gain=4)

        self.not_training = [self.conv1a, self.b2, self.b2_1, self.b2_2]

        self.from_scratch_layers = [self.f8_3, self.f8_4, self.f8_5, self.f9]

        self.predefined_featuresize = int(448//8)
        self.radius = 5
        self.ind_from, self.ind_to = pyutils.get_indices_of_pairs(radius=self.radius, size=(self.predefined_featuresize, self.predefined_featuresize))
        self.ind_from = torch.from_numpy(self.ind_from); self.ind_to = torch.from_numpy(self.ind_to)
        return

    def forward(self, x, mask, to_dense=False, stage=None):

        d = super().forward_as_dict(x)

        f8_3 = F.elu(self.f8_3(d['conv4']))
        f8_4 = F.elu(self.f8_4(d['conv5']))
        f8_5 = F.elu(self.f8_5(d['conv6']))
        x = F.elu(self.f9(torch.cat([f8_3, f8_4, f8_5], dim=1)))

        ## start add ##
        mask = F.interpolate(mask.float(), size=x.size()[2:], mode="nearest").long()
        mask = mask.view(mask.size(0), mask.size(1), -1).squeeze(0).squeeze(0).contiguous()
        ##  end  add ##

        if x.size(2) == self.predefined_featuresize and x.size(3) == self.predefined_featuresize:
            ind_from = self.ind_from
            ind_to = self.ind_to
        else:
            min_edge = min(x.size(2), x.size(3))
            radius = (min_edge-1)//2 if min_edge < self.radius*2+1 else self.radius
            ind_from, ind_to = pyutils.get_indices_of_pairs(radius, (x.size(2), x.size(3)))
            ind_from = torch.from_numpy(ind_from); ind_to = torch.from_numpy(ind_to)

        x = x.view(x.size(0), x.size(1), -1).contiguous()
        ind_from = ind_from.contiguous()
        ind_to = ind_to.contiguous()

        '''
        ff = torch.index_select(x, dim=2, index=ind_from.cuda(non_blocking=True))
        ft = torch.index_select(x, dim=2, index=ind_to.cuda(non_blocking=True))

        ff = torch.unsqueeze(ff, dim=2)
        ft = ft.view(ft.size(0), ft.size(1), -1, ff.size(3))

        aff = torch.exp(-torch.mean(torch.abs(ft-ff), dim=1))

        if to_dense:
            aff = aff.view(-1).cpu()

            ind_from_exp = torch.unsqueeze(ind_from, dim=0).expand(ft.size(2), -1).contiguous().view(-1)
            indices = torch.stack([ind_from_exp, ind_to])
            indices_tp = torch.stack([ind_to, ind_from_exp])

            area = x.size(2)
            indices_id = torch.stack([torch.arange(0, area).long(), torch.arange(0, area).long()])

            aff_mat = sparse.FloatTensor(torch.cat([indices, indices_id, indices_tp], dim=1),
                                      torch.cat([aff, torch.ones([area]), aff])).to_dense().cuda()

            return aff_mat

        else:
            assert 0
            return aff
        '''
        assert(to_dense == True and stage in ['first', 'second'])

        search_dist_size = ind_to.shape[0] // ind_from.shape[0]
        assert (ind_to.shape[0] % ind_from.shape[0] == 0)
        ind_from_exp = torch.unsqueeze(ind_from, dim=0).expand(search_dist_size, -1).contiguous().view(-1)

        if stage == 'first':
            valid_from = mask[ind_from_exp]
            valid_to = mask[ind_to]
            valid_final = valid_from * valid_to
            valid_index = torch.where(valid_final==1)[0]
            ind_from_exp = torch.index_select(ind_from_exp, dim=0, index=valid_index)
            ind_to = torch.index_select(ind_to, dim=0, index=valid_index)

            ff = torch.index_select(x, dim=2, index=ind_from_exp.cuda(non_blocking=True))
            ft = torch.index_select(x, dim=2, index=ind_to.cuda(non_blocking=True))

            aff = torch.exp(-torch.mean(torch.abs(ft-ff), dim=1))
            aff = aff.view(-1).cpu()

            indices = torch.stack([ind_from_exp, ind_to])
            indices_tp = torch.stack([ind_to, ind_from_exp])

            area = x.size(2)
            indices_id = torch.stack([torch.arange(0, area).long(), torch.arange(0, area).long()])

            aff_mat = sparse.FloatTensor(torch.cat([indices, indices_id, indices_tp], dim=1),
                                    torch.cat([aff, torch.ones([area]), aff])).to_dense().cuda()

        elif stage == 'second':
            valid_from = mask[ind_from_exp]
            valid_to = mask[ind_to]

            # 1: (boundary, boundary)
            valid_final_1 = (1-valid_from) * (1-valid_to)
            valid_index_1 = torch.where(valid_final_1==1)[0]
            ind_from_exp_1 = torch.index_select(ind_from_exp, dim=0, index=valid_index_1)
            ind_to_1 = torch.index_select(ind_to, dim=0, index=valid_index_1)

            ff_1 = torch.index_select(x, dim=2, index=ind_from_exp_1.cuda(non_blocking=True))
            ft_1 = torch.index_select(x, dim=2, index=ind_to_1.cuda(non_blocking=True))

            aff_1 = torch.exp(-torch.mean(torch.abs(ft_1-ff_1), dim=1))
            aff_1 = aff_1.view(-1).cpu()

            indices_1 = torch.stack([ind_from_exp_1, ind_to_1])
            indices_tp_1 = torch.stack([ind_to_1, ind_from_exp_1])

            # 2: nonboundary -> boundary
            valid_final_2 = valid_from * (1-valid_to)
            valid_index_2 = torch.where(valid_final_2==1)[0]
            ind_from_exp_2 = torch.index_select(ind_from_exp, dim=0, index=valid_index_2)
            ind_to_2 = torch.index_select(ind_to, dim=0, index=valid_index_2)

            ff_2 = torch.index_select(x, dim=2, index=ind_from_exp_2.cuda(non_blocking=True))
            ft_2 = torch.index_select(x, dim=2, index=ind_to_2.cuda(non_blocking=True))

            aff_2 = torch.exp(-torch.mean(torch.abs(ft_2-ff_2), dim=1))
            aff_2 = aff_2.view(-1).cpu()

            indices_2 = torch.stack([ind_from_exp_2, ind_to_2])

            # 3: boundary <- nonboundary
            valid_final_3 = (1-valid_from) * valid_to
            valid_index_3 = torch.where(valid_final_3==1)[0]
            ind_from_exp_3 = torch.index_select(ind_from_exp, dim=0, index=valid_index_3)
            ind_to_3 = torch.index_select(ind_to, dim=0, index=valid_index_3)

            ff_3 = torch.index_select(x, dim=2, index=ind_from_exp_3.cuda(non_blocking=True))
            ft_3 = torch.index_select(x, dim=2, index=ind_to_3.cuda(non_blocking=True))

            aff_3 = torch.exp(-torch.mean(torch.abs(ft_3-ff_3), dim=1))
            aff_3 = aff_3.view(-1).cpu()

            indices_tp_3 = torch.stack([ind_to_3, ind_from_exp_3])

            # 4: self
            area = x.size(2)
            indices_id = torch.stack([torch.arange(0, area).long(), torch.arange(0, area).long()])

            aff_mat = sparse.FloatTensor(torch.cat([indices_1, indices_tp_1, indices_2, indices_tp_3, indices_id], dim=1),
                                    torch.cat([aff_1, aff_1, aff_2, aff_3, torch.ones([area])])).to_dense().cuda()

        return aff_mat



    def get_parameter_groups(self):
        groups = ([], [], [], [])

        for m in self.modules():

            if (isinstance(m, nn.Conv2d) or isinstance(m, nn.modules.normalization.GroupNorm)):

                if m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:

                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)

        return groups



