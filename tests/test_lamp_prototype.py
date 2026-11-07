import math
import tempfile
import unittest
from pathlib import Path
import torch
from torch_geometric.utils import remove_self_loops, to_undirected
from Our_experiment.HCSAC.lamp_method.upstream_adapter import LAMPAdapter
from Our_experiment.HCSAC.lamp_method.lifecycle_lamp import LifecycleLAMP
from Our_experiment.HCSAC.lamp_method.critics import JointCriticHead, event_category, event_critic_loss, marginal_value
from Our_experiment.HCSAC.lamp_method.math_checks import decompose, run_checks
from Our_experiment.HCSAC.lamp_method.sequence_replay import SequenceReplay


def dense_reference(model, x, edges, weights, hidden):
    """Dense reconstruction of upstream to_undirected/normalization/message flow."""
    edges, weights = remove_self_loops(edges, weights)
    edges, weights = to_undirected(edges, weights, num_nodes=x.size(0))
    if weights is None: weights = torch.ones(edges.size(1), device=x.device)
    n = x.size(0); adj = x.new_zeros((n,n)); degree = x.new_zeros(n)
    for k in range(edges.size(1)):
        i,j = (int(edges[0,k]), int(edges[1,k])); adj[i,j] += weights[k]; degree[i] += weights[k]
    degree = degree.float(); inv = torch.where(degree>0, 1/degree, 0); sqrt = torch.sqrt(inv)
    sym = -sqrt[:,None]*adj*sqrt[None,:]
    anti = -inv[:,None]*adj + adj*inv[None,:]
    h = (model.mu*x@model.U + hidden)/model.xi
    skew = model.W-model.W.T-torch.diag(torch.sigmoid(model.mixing.dissipativity.diagonal))
    for _ in range(model.L):
        flow = h@skew.T + sym.T@model.S.linear(h) + model.beta*anti.T@model.K.linear(h) + model.bias
        h = h + model.epsilon*torch.tanh(flow)
    return h


class LampParity(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model=LAMPAdapter(4, hidden_dim=8)
        self.x=torch.randn(5,4); self.h=torch.randn(5,8)
        self.edges=torch.tensor([[0,1,1,2,3,3],[1,0,2,1,3,2]],dtype=torch.long)
        self.weights=torch.tensor([0.2,0.2,0.7,0.7,1.0,0.4])

    def test_dense_upstream_convention_parity(self):
        out,_=self.model.forward_snapshot(self.x,self.edges,self.weights,self.h,torch.arange(5))
        self.assertTrue(torch.allclose(out,dense_reference(self.model,self.x,self.edges,self.weights,self.h),atol=2e-6,rtol=2e-6))

    def test_empty_edges_and_trailing_isolated_nodes(self):
        edges=torch.empty((2,0),dtype=torch.long); w=torch.empty(0)
        out,_=self.model.forward_snapshot(self.x,edges,w,None,torch.arange(5))
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(torch.allclose(out,dense_reference(self.model,self.x,edges,w,torch.zeros(5,8)),atol=2e-6))

    def test_weights_affect_output(self):
        a,_=self.model.forward_snapshot(self.x,self.edges,self.weights,self.h,torch.arange(5))
        weights=self.weights.clone(); weights[2:4]*=4
        b,_=self.model.forward_snapshot(self.x,self.edges,weights,self.h,torch.arange(5))
        self.assertFalse(torch.allclose(a,b))

    def test_lifecycle_zero_init_matches_plain_and_gradients(self):
        other=LifecycleLAMP(4,19,hidden_dim=8)
        other.load_state_dict(self.model.state_dict(),strict=False)
        context=torch.randn(5,19);context[:,0]=1
        a,_=self.model.forward_snapshot(self.x,self.edges,self.weights,self.h,torch.arange(5))
        b,_=other.forward_snapshot(self.x,self.edges,self.weights,self.h,torch.arange(5),context)
        self.assertTrue(torch.allclose(a,b,atol=2e-6,rtol=2e-6))
        b.square().sum().backward()
        self.assertGreater(other.conditioner[-1].weight.grad.abs().sum().item(),0)
        with torch.no_grad():other.conditioner[-1].weight.add_(0.1)
        self.assertFalse(torch.allclose(other.damping(context,5)[0],other.damping(context,5)[1]))
        self.assertTrue(((other.damping(context,5)>0)&(other.damping(context,5)<1)).all())

    def test_node_permutation_and_history(self):
        p=torch.tensor([3,1,4,0,2]);inv=torch.argsort(p)
        a,_=self.model.forward_snapshot(self.x,self.edges,self.weights,self.h,torch.arange(5))
        b,_=self.model.forward_snapshot(self.x[p],inv[self.edges],self.weights,self.h[p],torch.arange(5)[p])
        self.assertTrue(torch.allclose(a[p],b,atol=2e-6))
        c,_=self.model.forward_snapshot(self.x,self.edges,self.weights,torch.zeros_like(self.h),torch.arange(5))
        self.assertFalse(torch.allclose(a,c))

    def test_zero_initialization_dtype(self):
        self.model.double()
        out,_=self.model.forward_snapshot(self.x.double(),self.edges,self.weights.double(),None,torch.arange(5))
        self.assertEqual(out.dtype,torch.float64)


class CriticMath(unittest.TestCase):
    def test_appledouble_files_excluded_from_source_fingerprint(self):
        from Our_experiment.HCSAC.lamp_method.run import python_source_paths
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'a.py').write_text('a=1\n');(p/'._a.py').write_bytes(b'not Python source')
            self.assertEqual([x.name for x in python_source_paths(p)],['a.py'])

    def test_math_identities_and_temperature(self):
        self.assertTrue(run_checks()['passed'])

    def test_event_mask_and_no_duplicates(self):
        self.assertEqual(event_category([{'event_type':'standby_to_active'},{'event_type':'reserve_launched'},
                                         {'event_type':'active_to_returning'},{'event_type':'charging_to_standby'}]),7)

    def test_probabilities_and_minimum_order(self):
        logits=torch.tensor([[0.,0.] + [-100.]*6])
        q1=torch.tensor([[0.,10.] + [0.]*6]);q2=torch.tensor([[10.,0.] + [0.]*6])
        self.assertAlmostEqual(marginal_value(logits,q1).item(),5.)
        self.assertAlmostEqual(torch.minimum(marginal_value(logits,q1),marginal_value(logits,q2)).item(),5.)
        self.assertNotAlmostEqual(marginal_value(logits,torch.minimum(q1,q2)).item(),5.)

    def test_ecc_loss_backpropagates(self):
        logits=torch.randn(3,8,requires_grad=True); values=torch.randn(3,8,requires_grad=True)
        loss=event_critic_loss(logits,values,torch.tensor([0,2,7]),torch.tensor([1.,2.,3.]))
        loss.backward();self.assertTrue(torch.isfinite(logits.grad).all() and torch.isfinite(values.grad).all())

    def test_conditional_expectation_and_error_bound(self):
        p=torch.tensor([.25,.75],dtype=torch.float64); q=torch.tensor([1.,3.],dtype=torch.float64)
        # Two-event one-step MDP: Y is 1 or 3 with probabilities .25 and .75.
        self.assertAlmostEqual(float((p*q).sum()),2.5)
        ph=torch.tensor([.3,.7],dtype=torch.float64); qh=torch.tensor([1.1,2.9],dtype=torch.float64)
        eps_q=(qh-q).abs().max();eps_p=(ph-p).abs().sum();bound=eps_q+q.abs().max()*eps_p
        self.assertLessEqual(float(((ph*qh).sum()-(p*q).sum()).abs()),float(bound))

    def test_inactive_action_cannot_change_q(self):
        head=JointCriticHead(8,True);h=torch.randn(10,8); mask=torch.tensor([1,0,0,0,0,0],dtype=torch.bool)
        a=torch.tensor([1,0,0,0,0,0]);b=torch.tensor([1,4,3,2,1,4])
        self.assertTrue(torch.equal(head(h,a,mask)[0],head(h,b,mask)[0]))

    def test_episode_boundaries(self):
        replay=SequenceReplay()
        with self.assertRaises(ValueError): replay.add_episode([{'terminal':True},{'terminal':True}])
        replay.add_episode([{'terminal':False},{'terminal':True}])
        for prefix,seq,sequence_mask in replay.sample(16,4):
            self.assertLessEqual(len(prefix)+len(seq),2)
            self.assertTrue(sequence_mask.all());self.assertEqual(len(sequence_mask),len(seq))


if __name__=='__main__': unittest.main()
