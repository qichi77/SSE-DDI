import math

import dgl.function as fn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.nn.inits import glorot
from torch_geometric.utils import softmax
from torch_scatter import scatter


def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: edges.src[src_field] * edges.dst[dst_field]}

    return func


def scaling(field, scale_constant):
    def func(edges):
        return {field: edges.data[field] / scale_constant}

    return func


def imp_exp_attn(implicit_attn, explicit_edge):
    """
    Multiply implicit node-derived attention scores by explicit edge features.

    Args:
        implicit_attn: Name of the implicit attention field.
        explicit_edge: Name of the explicit edge-feature field.
    """

    def func(edges):
        return {
            implicit_attn: (
                edges.data[implicit_attn] * edges.data[explicit_edge]
            )
        }

    return func


def out_edge_features(edge_feat):
    def func(edges):
        return {"e_out": edges.data[edge_feat]}

    return func


def exp(field):
    def func(edges):
        return {
            field: torch.exp(
                edges.data[field]
                .sum(-1, keepdim=True)
                .clamp(-5, 5)
            )
        }

    return func


class PairwiseGatedLineGraphConv(nn.Module):
    """Neighbor-specific gated message passing on a directed line graph."""

    def __init__(
        self,
        in_dim,
        out_dim=None,
        att_dim=None,
        gate_hidden_dim=None,
        eps=1e-8,
        collect_routes=False,
    ):
        super().__init__()

        if out_dim is None:
            out_dim = in_dim
        if att_dim is None:
            att_dim = out_dim
        if gate_hidden_dim is None:
            gate_hidden_dim = in_dim

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.att_dim = att_dim
        self.eps = eps
        self.collect_routes = bool(collect_routes)

        # The destination bond state supplies the query.
        self.query = nn.Linear(in_dim, att_dim)

        # The source bond state supplies the key and value.
        self.key = nn.Linear(in_dim, att_dim)
        self.value = nn.Linear(in_dim, out_dim)

        # Pair-conditioned transition gate:
        # [h_src || h_dst || h_src * h_dst] -> scalar.
        self.gate_mlp = nn.Sequential(
            nn.Linear(3 * in_dim, gate_hidden_dim),
            nn.PReLU(),
            nn.Linear(gate_hidden_dim, 1),
        )

        # Optional visualization outputs.
        self.att_weights = None
        self.gate_values = None
        self.route_edge_index = None

    def set_collect_routes(self, enabled=True):
        self.collect_routes = bool(enabled)

        if not self.collect_routes:
            self.att_weights = None
            self.gate_values = None
            self.route_edge_index = None

    def forward(self, x, edge_index):
        """
        Args:
            x:
                Directed bond-state features with shape
                [num_bond_states, in_dim].
            edge_index:
                Directed line-graph transitions with shape [2, num_routes].
                edge_index[0] is the source state and edge_index[1] is the
                destination state.

        Returns:
            Selectively aggregated incoming messages with shape
            [num_bond_states, out_dim].
        """
        num_nodes = x.size(0)

        # Molecules such as a single isolated bond may have no valid
        # non-backtracking line-graph transition.
        if edge_index.numel() == 0:
            if self.collect_routes:
                self.att_weights = x.new_empty((0,)).detach().cpu()
                self.gate_values = x.new_empty((0,)).detach().cpu()
                self.route_edge_index = edge_index.detach().cpu()
            else:
                self.att_weights = None
                self.gate_values = None
                self.route_edge_index = None

            return x.new_zeros((num_nodes, self.out_dim))

        src, dst = edge_index

        q = self.query(x)  # [N, d_a]
        k = self.key(x)    # [N, d_a]
        v = self.value(x)  # [N, out_dim]

        pair_feature = torch.cat(
            [
                x[src],
                x[dst],
                x[src] * x[dst],
            ],
            dim=-1,
        )  # [E_line, 3 * in_dim]

        gate = torch.sigmoid(
            self.gate_mlp(pair_feature)
        ).squeeze(-1)  # [E_line]

        score = (
            q[dst] * k[src]
        ).sum(dim=-1) / math.sqrt(self.att_dim)  # [E_line]

        # exp(score + log(gate)) = gate * exp(score)
        gated_logit = score + torch.log(
            gate.clamp_min(self.eps)
        )

        # Normalize over the incoming transitions of each destination state.
        alpha = softmax(
            gated_logit,
            dst,
            num_nodes=num_nodes,
        )  # [E_line]

        message = alpha.unsqueeze(-1) * v[src]

        out = scatter(
            message,
            dst,
            dim=0,
            dim_size=num_nodes,
            reduce="sum",
        )  # [N, out_dim]

        if self.collect_routes:
            # Detach and move visualization tensors to CPU so that they do
            # not retain autograd graphs or occupy GPU memory.
            self.att_weights = alpha.detach().cpu()
            self.gate_values = gate.detach().cpu()
            self.route_edge_index = edge_index.detach().cpu()
        else:
            self.att_weights = None
            self.gate_values = None
            self.route_edge_index = None

        return out


class SelfAttentionGlobalPool(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        collect_weights=False,
    ):
        super().__init__()

        self.att_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        self.collect_weights = bool(collect_weights)
        self.att_weights = None

    def set_collect_weights(self, enabled=True):
        self.collect_weights = bool(enabled)

        if not self.collect_weights:
            self.att_weights = None

    def forward(self, x, batch):
        scores = self.att_mlp(x)  # [N, 1]
        weights = softmax(scores, batch, dim=0)  # [N, 1]
        out = global_add_pool(x * weights, batch)  # [B, D]

        if self.collect_weights:
            self.att_weights = weights.detach().cpu()
        else:
            self.att_weights = None

        return out


class DMPNN(nn.Module):
    def __init__(
        self,
        edge_dim,
        n_feats,
        n_iter,
        route_dropout=0.1,
        gate_hidden_dim=None,
        collect_routes=False,
    ):
        super().__init__()

        self.n_iter = n_iter
        self.n_feats = n_feats
        self.collect_routes = bool(collect_routes)

        # Initial directed bond-state encoding.
        self.lin_u = nn.Linear(n_feats, n_feats, bias=False)
        self.lin_v = nn.Linear(n_feats, n_feats, bias=False)
        self.lin_edge = nn.Linear(edge_dim, n_feats, bias=False)

        # Neighbor-specific selective message router.
        self.route_conv = PairwiseGatedLineGraphConv(
            in_dim=n_feats,
            out_dim=n_feats,
            att_dim=n_feats,
            gate_hidden_dim=gate_hidden_dim,
            collect_routes=self.collect_routes,
        )

        # Selectively routed messages enter the recurrent state update.
        self.state_update = nn.GRUCell(
            input_size=n_feats,
            hidden_size=n_feats,
        )

        self.route_dropout = nn.Dropout(route_dropout)

        # Separate normalization for each propagation iteration.
        self.state_norms = nn.ModuleList(
            [nn.LayerNorm(n_feats) for _ in range(n_iter)]
        )

        # Graph-level readout does not perform another line-graph routing.
        self.readout = SelfAttentionGlobalPool(
            input_dim=n_feats,
            hidden_dim=n_feats,
            collect_weights=self.collect_routes,
        )

        # Iteration-wise weighting.
        self.a = nn.Parameter(
            torch.zeros(1, n_feats, n_iter)
        )
        self.a_bias = nn.Parameter(
            torch.zeros(1, 1, n_iter)
        )

        self.lin_gout = nn.Linear(n_feats, n_feats)
        glorot(self.a)

        self.lin_block = LinearBlock(n_feats)

        # Optional visualization histories for the current encoder call.
        self.route_gate_history = []
        self.route_attention_history = []
        self.pool_attention_history = []

    def set_collect_routes(self, enabled=True):
        self.collect_routes = bool(enabled)

        self.route_conv.set_collect_routes(
            self.collect_routes
        )
        self.readout.set_collect_weights(
            self.collect_routes
        )

        if not self.collect_routes:
            self.route_gate_history.clear()
            self.route_attention_history.clear()
            self.pool_attention_history.clear()

    def forward(self, data, atom_x):
        edge_index = data.edge_index
        line_edge_index = data.line_graph_edge_index
        bond_batch = data.edge_index_batch

        edge_u = self.lin_u(atom_x)
        edge_v = self.lin_v(atom_x)
        edge_uv = self.lin_edge(data.edge_attr)

        edge_attr = (
            edge_u[edge_index[0]]
            + edge_v[edge_index[1]]
            + edge_uv
        ) / 3.0

        # h^(0)
        state = edge_attr

        state_list = []
        graph_state_list = []

        # Clear tensors saved by the previous encoder call.
        self.route_gate_history.clear()
        self.route_attention_history.clear()
        self.pool_attention_history.clear()

        for step in range(self.n_iter):
            routed_message = self.route_conv(
                state,
                line_edge_index,
            )

            # Dropout regularizes only newly routed neighbor information;
            # the initial bond-state residual remains intact.
            routed_message = self.route_dropout(
                routed_message
            )

            candidate = edge_attr + routed_message

            state = self.state_update(
                candidate,
                state,
            )
            state = self.state_norms[step](state)

            state_list.append(state)

            graph_state = self.readout(
                state,
                bond_batch,
            )
            graph_state = torch.tanh(
                self.lin_gout(graph_state)
            )
            graph_state_list.append(graph_state)

            if self.collect_routes:
                self.route_gate_history.append(
                    self.route_conv.gate_values
                )
                self.route_attention_history.append(
                    self.route_conv.att_weights
                )
                self.pool_attention_history.append(
                    self.readout.att_weights
                )

        graph_state_all = torch.stack(
            graph_state_list,
            dim=-1,
        )  # [B, D, K]

        state_all = torch.stack(
            state_list,
            dim=-1,
        )  # [E_bond, D, K]

        iteration_scores = (
            graph_state_all * self.a
        ).sum(dim=1, keepdim=True) + self.a_bias

        iteration_weights = torch.softmax(
            iteration_scores,
            dim=-1,
        )  # [B, 1, K]

        beta_edge = (
            iteration_weights
            .squeeze(1)[bond_batch]
            .unsqueeze(1)
        )  # [E_bond, 1, K]

        # Weighted multi-depth bond-state condensation.
        state = (
            state_all * beta_edge
        ).sum(dim=-1)  # [E_bond, D]

        incoming_bond_sum = scatter(
            state,
            edge_index[1],
            dim=0,
            dim_size=atom_x.size(0),
            reduce="sum",
        )

        atom_state = atom_x + incoming_bond_sum
        atom_state = self.lin_block(atom_state)

        return atom_state


class LinearBlock(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.snd_n_feats = 6 * n_feats

        self.lin1 = nn.Sequential(
            nn.BatchNorm1d(n_feats),
            nn.Linear(n_feats, self.snd_n_feats),
        )
        self.lin2 = nn.Sequential(
            nn.BatchNorm1d(self.snd_n_feats),
            nn.PReLU(),
            nn.Linear(self.snd_n_feats, self.snd_n_feats),
        )
        self.lin3 = nn.Sequential(
            nn.BatchNorm1d(self.snd_n_feats),
            nn.PReLU(),
            nn.Linear(self.snd_n_feats, self.snd_n_feats),
        )
        self.lin4 = nn.Sequential(
            nn.BatchNorm1d(self.snd_n_feats),
            nn.PReLU(),
            nn.Linear(self.snd_n_feats, self.snd_n_feats),
        )
        self.lin5 = nn.Sequential(
            nn.BatchNorm1d(self.snd_n_feats),
            nn.PReLU(),
            nn.Linear(self.snd_n_feats, n_feats),
        )

    def forward(self, x):
        x = self.lin1(x)
        x = (self.lin3(self.lin2(x)) + x) / 2
        x = (self.lin4(x) + x) / 2
        x = self.lin5(x)
        return x


class DrugEncoder(nn.Module):
    def __init__(
        self,
        in_dim,
        edge_in_dim,
        hidden_dim,
        n_iter,
        collect_routes=False,
    ):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        )

        self.line_graph = DMPNN(
            edge_dim=edge_in_dim,
            n_feats=hidden_dim,
            n_iter=n_iter,
            collect_routes=collect_routes,
        )

    def forward(self, data):
        # Keep data.x unchanged so that the same Data/Batch object can be reused.
        atom_x = self.mlp(data.x)
        return self.line_graph(
            data=data,
            atom_x=atom_x,
        )


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, use_bias):
        super().__init__()

        self.out_dim = out_dim
        self.num_heads = num_heads

        if use_bias:
            self.Q = nn.Linear(
                in_dim,
                out_dim * num_heads,
                bias=True,
            )
            self.K = nn.Linear(
                in_dim,
                out_dim * num_heads,
                bias=True,
            )
            self.V = nn.Linear(
                in_dim,
                out_dim * num_heads,
                bias=True,
            )
            self.proj_e = nn.Linear(
                in_dim,
                out_dim * num_heads,
                bias=True,
            )
        else:
            self.Q = nn.Linear(
                in_dim,
                out_dim * num_heads,
                bias=False,
            )
            self.K = nn.Linear(
                in_dim,
                out_dim * num_heads,
                bias=False,
            )
            self.V = nn.Linear(
                in_dim,
                out_dim * num_heads,
                bias=False,
            )
            self.proj_e = nn.Linear(
                in_dim,
                out_dim * num_heads,
                bias=False,
            )

    def propagate_attention(self, g):
        # Compute implicit attention score.
        g.apply_edges(
            src_dot_dst("K_h", "Q_h", "score")
        )

        # Scaled dot-product attention.
        g.apply_edges(
            scaling("score", np.sqrt(self.out_dim))
        )

        # Use explicit edge features to modulate the scores.
        g.apply_edges(
            imp_exp_attn("score", "proj_e")
        )

        # Copy edge features for the edge FFN.
        g.apply_edges(
            out_edge_features("score")
        )

        # Exponentiate scores.
        g.apply_edges(exp("score"))

        # Send weighted values to target nodes.
        eids = g.edges()
        g.send_and_recv(
            eids,
            fn.src_mul_edge("V_h", "score", "V_h"),
            fn.sum("V_h", "wV"),
        )
        g.send_and_recv(
            eids,
            fn.copy_edge("score", "score"),
            fn.sum("score", "z"),
        )

    def forward(self, g, h, e):
        q_h = self.Q(h)
        k_h = self.K(h)
        v_h = self.V(h)
        proj_e = self.proj_e(e)

        g.ndata["Q_h"] = q_h.view(
            -1,
            self.num_heads,
            self.out_dim,
        )
        g.ndata["K_h"] = k_h.view(
            -1,
            self.num_heads,
            self.out_dim,
        )
        g.ndata["V_h"] = v_h.view(
            -1,
            self.num_heads,
            self.out_dim,
        )
        g.edata["proj_e"] = proj_e.view(
            -1,
            self.num_heads,
            self.out_dim,
        )

        self.propagate_attention(g)

        h_out = g.ndata["wV"] / (
            g.ndata["z"]
            + torch.full_like(g.ndata["z"], 1e-6)
        )
        e_out = g.edata["e_out"]

        return h_out, e_out


class GraphTransformerLayer(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        num_heads,
        dropout=0.0,
        layer_norm=False,
        batch_norm=True,
        residual=True,
        use_bias=False,
    ):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm

        self.attention = MultiHeadAttentionLayer(
            in_dim,
            out_dim // num_heads,
            num_heads,
            use_bias,
        )

        self.O_h = nn.Linear(out_dim, out_dim)
        self.O_e = nn.Linear(out_dim, out_dim)

        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(out_dim)
            self.layer_norm1_e = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm1_h = nn.BatchNorm1d(out_dim)
            self.batch_norm1_e = nn.BatchNorm1d(out_dim)

        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_h_layer2 = nn.Linear(out_dim * 2, out_dim)

        self.FFN_e_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_e_layer2 = nn.Linear(out_dim * 2, out_dim)

        if self.layer_norm:
            self.layer_norm2_h = nn.LayerNorm(out_dim)
            self.layer_norm2_e = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm2_h = nn.BatchNorm1d(out_dim)
            self.batch_norm2_e = nn.BatchNorm1d(out_dim)

    def forward(self, g, h, e):
        if self.layer_norm:
            h = self.layer_norm1_h(h)
            e = self.layer_norm1_e(e)

        if self.batch_norm:
            h = self.batch_norm1_h(h)
            e = self.batch_norm1_e(e)

        h_in1 = h
        e_in1 = e

        h_attn_out, e_attn_out = self.attention(g, h, e)
        h = h_attn_out.view(-1, self.out_channels)
        e = e_attn_out.view(-1, self.out_channels)

        h = F.dropout(h, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)

        h = self.O_h(h)
        e = self.O_e(e)

        if self.residual:
            h = h_in1 + h
            e = e_in1 + e

        if self.layer_norm:
            h = self.layer_norm2_h(h)
            e = self.layer_norm2_e(e)

        if self.batch_norm:
            h = self.batch_norm2_h(h)
            e = self.batch_norm2_e(e)

        h_in2 = h
        e_in2 = e

        h = self.FFN_h_layer1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)

        e = self.FFN_e_layer1(e)
        e = F.relu(e)
        e = F.dropout(e, self.dropout, training=self.training)
        e = self.FFN_e_layer2(e)

        if self.residual:
            h = h_in2 + h
            e = e_in2 + e

        return h, e

    def __repr__(self):
        return (
            "{}(in_channels={}, out_channels={}, heads={}, residual={})"
            .format(
                self.__class__.__name__,
                self.in_channels,
                self.out_channels,
                self.num_heads,
                self.residual,
            )
        )


class GraphTransformerNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()

        num_atom_type = net_params["num_atom_type"]
        num_bond_type = net_params["num_bond_type"]
        hidden_dim = net_params["hidden_dim"]
        num_heads = net_params["n_heads"]
        out_dim = net_params["out_dim"]
        in_feat_dropout = net_params["in_feat_dropout"]
        dropout = net_params["dropout"]
        n_layers = net_params["L"]

        self.device = net_params["device"]
        self.readout = net_params["readout"]
        self.layer_norm = net_params["layer_norm"]
        self.batch_norm = net_params["batch_norm"]
        self.residual = net_params["residual"]
        self.edge_feat = net_params["edge_feat"]
        self.lap_pos_enc = net_params["lap_pos_enc"]

        in_dim = net_params["num_atom_type"]
        edge_in_dim = net_params["num_bond_type"]
        n_iter = net_params["n_iter"]

        num_relations = int(net_params["num_relations"])
        sim_dim = int(net_params["sim_dim"])
        collect_routes = bool(
            net_params.get("collect_routes", False)
        )

        # The current residual, normalization, and fusion implementation
        # assumes that every Transformer layer preserves the hidden width.
        if num_heads < 1:
            raise ValueError("n_heads must be at least 1.")

        if n_layers < 1:
            raise ValueError("L must be at least 1.")

        if hidden_dim % num_heads != 0:
            raise ValueError(
                "hidden_dim must be divisible by n_heads. "
                f"Received hidden_dim={hidden_dim}, "
                f"n_heads={num_heads}."
            )

        if out_dim % num_heads != 0:
            raise ValueError(
                "out_dim must be divisible by n_heads. "
                f"Received out_dim={out_dim}, "
                f"n_heads={num_heads}."
            )

        if out_dim != hidden_dim:
            raise ValueError(
                "The current GraphTransformer residual and fusion "
                "implementation requires out_dim == hidden_dim. "
                f"Received out_dim={out_dim}, "
                f"hidden_dim={hidden_dim}."
            )

        if num_relations < 1:
            raise ValueError(
                "num_relations must be a positive integer."
            )

        if sim_dim < 1:
            raise ValueError(
                "sim_dim must be a positive integer."
            )

        self.drug_encoder = DrugEncoder(
            in_dim=in_dim,
            edge_in_dim=edge_in_dim,
            hidden_dim=hidden_dim,
            n_iter=n_iter,
            collect_routes=collect_routes,
        )

        if self.edge_feat:
            self.embedding_e = nn.Linear(
                num_bond_type,
                hidden_dim,
            )
        else:
            self.embedding_e = nn.Linear(
                1,
                hidden_dim,
            )

        self.lin = nn.Sequential(
            nn.Linear(hidden_dim * 6, hidden_dim * 2),
            nn.PReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        self.graph_pred_linear = nn.Identity()

        self.rmodule = nn.Embedding(
            num_embeddings=num_relations,
            embedding_dim=hidden_dim,
        )

        self.in_feat_dropout = nn.Dropout(
            in_feat_dropout
        )

        self.layers = nn.ModuleList(
            [
                GraphTransformerLayer(
                    hidden_dim,
                    hidden_dim,
                    num_heads,
                    dropout,
                    self.layer_norm,
                    self.batch_norm,
                    self.residual,
                )
                for _ in range(n_layers - 1)
            ]
        )

        self.layers.append(
            GraphTransformerLayer(
                hidden_dim,
                out_dim,
                num_heads,
                dropout,
                self.layer_norm,
                self.batch_norm,
                self.residual,
            )
        )

        self.lin_sim = nn.Linear(
            sim_dim,
            hidden_dim,
        )

    def Fusion(self, sub, sim, data):
        max_pool = global_max_pool(sub, data.batch)
        mean_pool = global_mean_pool(sub, data.batch)
        d_g = torch.cat(
            [max_pool, mean_pool],
            dim=-1,
        ).type_as(sub)
        d_g = self.graph_pred_linear(d_g)

        sim = self.lin_sim(sim.float())
        global_graph = torch.cat(
            [d_g, sim],
            dim=-1,
        )
        return global_graph

    def forward(
        self,
        h_data,
        t_data,
        g1,
        g2,
        e1,
        e2,
        rel,
        sim1,
        sim2,
    ):
        s_h = self.drug_encoder(h_data)
        s_t = self.drug_encoder(t_data)

        h1 = self.in_feat_dropout(s_h)
        h2 = self.in_feat_dropout(s_t)

        e1 = self.embedding_e(e1.float())
        e2 = self.embedding_e(e2.float())

        for conv in self.layers:
            h1, e1 = conv(g1, h1, e1)
            h2, e2 = conv(g2, h2, e2)

        h = self.Fusion(h1, sim1, h_data)
        t = self.Fusion(h2, sim2, t_data)

        pair = torch.cat([h, t], dim=-1).float()
        rfeat = self.rmodule(rel)
        logit = (self.lin(pair) * rfeat).sum(-1)

        return logit


def GraphTransformer(net_params):
    return GraphTransformerNet(net_params)


def gnn_model(model_name, net_params):
    models = {
        "GraphTransformer": GraphTransformer,
    }
    return models[model_name](net_params)
