use vstd::prelude::*;
use crate::semantics::*;

verus! {

pub struct Permission {
    pub access: Map<Address, Seq<bool>>,
}

pub struct PermissionAugmentation {
    pub aug_map: Map<ChannelIndex, Seq<Permission>>,
}

/**
 * How many read permissions does a write permission splits into
 * Has to be a positive integer
 */
pub spec fn permission_write_split() -> nat;

#[verifier(external_body)]
#[verifier(broadcast_forall)]
pub proof fn axiom_permission_write_split_positive()
    ensures permission_write_split() > 0;

impl Permission {
    pub open spec fn valid(self) -> bool
    {
        (forall |addr: Address| self.access.dom().contains(addr)) &&
        (forall |addr: Address| (#[trigger] self.access[addr]).len() == permission_write_split())
    }

    /**
     * Condition for two permissions to be considered disjoint
     */
    pub open spec fn disjoint(self, other: Permission) -> bool
        recommends
            self.valid(),
            other.valid(),
    {
        forall |addr: Address, i: int|
            #![trigger self.access[addr][i]]
            #![trigger other.access[addr][i]]
            0 <= i < permission_write_split() ==>
            !(self.access[addr][i] && other.access[addr][i])
    }

    /**
     * (Disjoint) union of a list of permissions
     */
    pub open spec fn union_of(perms: Seq<Permission>) -> Permission
        recommends
            forall |i: int| 0 <= i < perms.len() ==> (#[trigger] perms[i]).valid(),

            // mutually disjoint
            forall |i: int, j: int| 0 <= i < perms.len() && 0 <= j < perms.len() && i != j ==> perms[i].disjoint(perms[j]),
    {
        Permission {
            access: Map::total(|addr: Address|
                Seq::new(permission_write_split(),
                    |i: int| exists |j: int| 0 <= j < perms.len() && #[trigger] perms[j].access[addr][i])),
        }
    }

    /**
     * Whether the permission has read permission on an address
     */
    pub open spec fn has_read(self, addr: Address) -> bool
        recommends self.valid()
    {
        exists |i: int| 0 <= i < permission_write_split() && self.access[addr][i]
    }

    /**
     * Whether the permission has write permission on an address
     */
    pub open spec fn has_write(self, addr: Address) -> bool
        recommends self.valid()
    {
        forall |i: int| 0 <= i < permission_write_split() ==> self.access[addr][i]
    }

    /**
     * Condition for self to contain other
     */
    pub open spec fn contains(self, other: Permission) -> bool
        recommends
            self.valid(),
            other.valid(),
    {
        forall |addr: Address, i: int|
            #![trigger other.access[addr][i]]
            #![trigger self.access[addr][i]]
            0 <= i < permission_write_split() ==>
            (other.access[addr][i] ==> self.access[addr][i])
    }
}

impl PermissionAugmentation {
    /**
     * Whether an augmentation is valid wrt a configuration
     */
    pub open spec fn valid(self, config: Configuration) -> bool
        recommends config.valid()
    {
        // Domain should be exactly the set of channels
        self.aug_map.dom() =~= config.graph.channels &&

        // Each permission should be valid
        (forall |channel: ChannelIndex|
            #[trigger] config.is_channel(channel) ==>
            self.aug_map[channel].len() == config.channels[channel].len() &&
            forall |i: int| 0 <= i < self.aug_map[channel].len() ==> (#[trigger] self.aug_map[channel][i]).valid()) &&

        // Permissions should be mutually disjoint
        (forall |channel1: ChannelIndex, channel2: ChannelIndex, i: int, j: int|
            config.is_channel(channel1) && config.is_channel(channel2) &&
            0 <= i < self.aug_map[channel1].len() && 0 <= j < self.aug_map[channel2].len() &&
            (channel1 != channel2 || i != j) ==>
            self.aug_map[channel1][i].disjoint(self.aug_map[channel2][j]))
    }

    pub open spec fn get_op_input_permissions(self, config: Configuration, op: OperatorIndex) -> Seq<Permission>
        recommends self.valid(config)
    {
        let inputs = config.get_op_input_channels(op);
        Seq::new(inputs.len(), |i: int| self.aug_map[inputs[i]].first())
    }
}

pub open spec fn consistent_step(
    op: OperatorIndex,
    config1: Configuration, aug1: PermissionAugmentation,
    config2: Configuration, aug2: PermissionAugmentation,
) -> bool
{
    let inputs = config1.get_op_input_channels(op);
    let outputs = config1.get_op_output_channels(op);

    let input_perms = Seq::new(inputs.len(), |i: int| aug1.aug_map[inputs[i]].first());
    let output_perms = Seq::new(outputs.len(), |i: int| aug2.aug_map[outputs[i]].last());

    config1.valid() &&
    config2.valid() &&
    config2 == config1.step(op) &&
    aug1.valid(config1) &&
    aug2.valid(config2) &&

    // Four cases:
    // 1. channel not in inputs or outputs
    // 2. channel in inputs only
    // 3. channel in outputs only
    // 4. channel in both inputs and outputs
    (forall |channel: ChannelIndex|
        #[trigger] config1.is_channel(channel) ==> {
            if !inputs.contains(channel) && !outputs.contains(channel) {
                // The permissions are unchanged in case 1
                aug2.aug_map[channel] =~= aug1.aug_map[channel]
            } else if inputs.contains(channel) && !outputs.contains(channel) {
                // Case 2
                // aug2.aug_map[channel].len() == aug1.aug_map[channel].len() - 1 &&
                aug2.aug_map[channel] =~= aug1.aug_map[channel].drop_first()
            } else if !inputs.contains(channel) && outputs.contains(channel) {
                // Case 3
                // aug2.aug_map[channel].len() == aug1.aug_map[channel].len() + 1 &&
                aug2.aug_map[channel].drop_last() =~= aug1.aug_map[channel]
            } else {
                // Case 4
                // aug2.aug_map[channel].len() == aug1.aug_map[channel].len() &&
                aug2.aug_map[channel].drop_last() =~= aug1.aug_map[channel].drop_first()
            }
        }
    ) &&

    // Union of input perms is less than equal to the union of output perms
    Permission::union_of(input_perms).contains(Permission::union_of(output_perms)) &&

    // If the operator is a read/write, we require suitable permissions
    (config1.operators[op].is_Read() ==>
        Permission::union_of(input_perms).has_read(config1.get_op_input_values(op)[0].as_address())) &&
        
    (config1.operators[op].is_Write() ==>
        Permission::union_of(input_perms).has_write(config1.get_op_input_values(op)[0].as_address()))
}

/**
 * Lemma: Similar to lemma_step_independence, but for input permissions
 */
pub proof fn lemma_step_independent_input_permissions(
    op1: OperatorIndex, op2: OperatorIndex,
    config1: Configuration, aug1: PermissionAugmentation,
    config2: Configuration, aug2: PermissionAugmentation,
)
    requires
        config1.fireable(op1),
        config1.fireable(op2),
        op1 != op2,
        consistent_step(op1, config1, aug1, config2, aug2),

    ensures
        aug1.get_op_input_permissions(config1, op2) == aug2.get_op_input_permissions(config2, op2)
{
    reveal(Configuration::step);
    assert(aug1.get_op_input_permissions(config1, op2) =~= aug2.get_op_input_permissions(config2, op2));
}

// Permission {
//     access: Map::total(|addr: Address|
//         Seq::new(permission_write_split(),
//             |i: int| exists |j: int| 0 <= j < perms.len() && #[trigger] perms[j].access[addr][i])),
// }

/**
 * Lemma: Let perms1 and perms2 be two lists of permissions.
 * If any permission in perms1 is disjoint from any permission
 * in perms2, then the union of perms1 is disjoint from the
 * union of perms2.
 */
proof fn lemma_mutually_disjoint_union(perms1: Seq<Permission>, perms2: Seq<Permission>)
    requires
        perms1.len() > 0,
        perms2.len() > 0,

        forall |i: int, j: int|
            #![trigger perms1[i], perms2[j]]
            0 <= i < perms1.len() && 0 <= j < perms2.len() ==>
            perms1[i].disjoint(perms2[j])

    ensures
        Permission::union_of(perms1).disjoint(Permission::union_of(perms2))
{}

/**
 * Lemma: If both **memory** operators op1 and op2 are fireable in a
 * configuration (and one of op1 and op2 is a write), and op1 and op2
 * can be fired in consistent steps, then their accessed memory addresses
 * must be different.
 */
proof fn lemma_consistent_step_disjoint_memory_address(
    op1: OperatorIndex, op2: OperatorIndex,
    config1: Configuration, aug1: PermissionAugmentation,
    config2: Configuration, aug2: PermissionAugmentation,
    config3: Configuration, aug3: PermissionAugmentation,
)
    requires
        config1.valid(),
        config2.valid(),
        config3.valid(),
        config1.fireable(op1),
        config1.fireable(op2),
        op1 != op2,

        // config1 -> config2 -> config3 is a consistent trace
        consistent_step(op1, config1, aug1, config2, aug2),
        consistent_step(op2, config2, aug2, config3, aug3),

        // op1 and op2 are memory operators, and at least one of op1 and op2 is a write
        config1.operators[op1].is_Read() || config1.operators[op1].is_Write(),
        config1.operators[op2].is_Read() || config1.operators[op2].is_Write(),
        config1.operators[op1].is_Write() || config1.operators[op2].is_Write(),

    ensures
        config1.get_op_input_values(op1)[0].as_address() != config1.get_op_input_values(op2)[0].as_address()
{
    config1.lemma_step_independence(op1, op2);

    let op1_inputs = config1.get_op_input_channels(op1);
    let op2_inputs = config1.get_op_input_channels(op2);

    let op1_address = config1.get_op_input_values(op1)[0].as_address();
    let op2_address = config1.get_op_input_values(op2)[0].as_address();

    let op1_input_perms_init = aug1.get_op_input_permissions(config1, op1);

    let op2_input_perms_init = aug1.get_op_input_permissions(config1, op2);
    let op2_input_perms_after_op1 = aug2.get_op_input_permissions(config2, op2);

    lemma_step_independent_input_permissions(op1, op2, config1, aug1, config2, aug2);
    assert(op2_input_perms_init =~= op2_input_perms_after_op1);

    assert(op1_address != op2_address) by {
        let op1_input_perm_union = Permission::union_of(op1_input_perms_init);
        let op2_input_perm_union = Permission::union_of(op2_input_perms_init);

        assert(op1_input_perm_union.disjoint(op2_input_perm_union)) by {
            lemma_mutually_disjoint_union(op1_input_perms_init, op2_input_perms_init);
        }

        // Read/write operator type of op2 is preserved after op1 is fired
        if (config1.operators[op2].is_Read()) {
            assert(config2.operators[op2].is_Read()) by {
                reveal(Configuration::step);
            }
        } else if (config1.operators[op2].is_Write()) {
            assert(config2.operators[op2].is_Write()) by {
                reveal(Configuration::step);
            }
        }

        // Proof by contradiction
        if op1_address == op2_address {
            if (config1.operators[op1].is_Write() && config1.operators[op2].is_Write()) {
                // assert(op1_input_perm_union.has_write(op1_address));
                // assert(op2_input_perm_union.has_write(op2_address));

                // assert forall |i: int|
                //     0 <= i < permission_write_split()
                // implies
                //     op1_input_perm_union.access[op1_address][i] && false by
                // {
                //     assert(op1_input_perm_union.access[op1_address][i]);
                //     assert(op2_input_perm_union.access[op1_address][i]);
                //     assert(!(op1_input_perm_union.access[op1_address][i] && op2_input_perm_union.access[op1_address][i]));
                // }

                let trigger = op1_input_perm_union.access[op1_address][0];
            }
            assert(false);
        }
    }
}

/**
 * Returns a valid augmentation aug4 for config1.step(op2)
 * such that (config1, aug1) -> (config1.step(op2), aug4) -> (config3, aug3)
 * is a consistent trace.
 */
spec fn consistent_step_commute_aug4_choice(
    op1: OperatorIndex, op2: OperatorIndex,
    config1: Configuration, aug1: PermissionAugmentation,
    config2: Configuration, aug2: PermissionAugmentation,
    config3: Configuration, aug3: PermissionAugmentation,
) -> PermissionAugmentation
{
    let inputs = config1.get_op_input_channels(op2);
    let outputs = config1.get_op_output_channels(op2);

    let input_perms = aug1.get_op_input_permissions(config1, op2);
    let output_perms = Seq::new(outputs.len(), |i: int| aug3.aug_map[outputs[i]].last());

    // let input_updated_aug_map = Map::new(
    //     |channel: ChannelIndex| config1.is_channel(channel),
    //     |channel: ChannelIndex|
    //         if inputs.contains(channel) { aug1.aug_map[channel].drop_first() }
    //         else { aug1.aug_map[channel] }
    // );

    // let output_updated_aug_map = Map::new(
    //     |channel: ChannelIndex| config1.is_channel(channel),
    //     |channel: ChannelIndex|
    //         if outputs.contains(channel) {
    //             let output_index = outputs.index_of(channel);
    //             input_updated_aug_map[channel].push(output_perms[output_index])
    //         }
    //         else { input_updated_aug_map[channel] }
    // );

    let new_aug_map = Map::new(
        |channel: ChannelIndex| config1.is_channel(channel),
        |channel: ChannelIndex|
            if !inputs.contains(channel) && !outputs.contains(channel) {
                aug1.aug_map[channel]
            } else if inputs.contains(channel) && !outputs.contains(channel) {
                aug1.aug_map[channel].drop_first()
            } else if !inputs.contains(channel) && outputs.contains(channel) {
                let output_index = outputs.index_of(channel);
                aug1.aug_map[channel].push(output_perms[output_index])
            } else {
                let output_index = outputs.index_of(channel);
                aug1.aug_map[channel].drop_first().push(output_perms[output_index])
            }
    );

    PermissionAugmentation { aug_map: new_aug_map }
}

// Some random seq fact
proof fn lemma_seq_drop_last<A>(seq1: Seq<A>, seq2: Seq<A>)
    requires
        seq1.len() > 0,
        seq2.len() > 0,
        seq1.drop_last() == seq2.drop_last(),
        seq1.last() == seq2.last(),

    ensures
        seq1 =~= seq2,
{
    assert(seq1.len() == seq1.drop_last().len() + 1);
    assert(seq2.len() == seq2.drop_last().len() + 1);

    assert forall |i: int|
        0 <= i < seq1.len()
    implies
        seq1[i] == seq2[i] by
    {
        if 0 <= i < seq1.drop_last().len() {
            assert(seq1[i] == seq1.drop_last()[i]);
            assert(seq2[i] == seq2.drop_last()[i]);
        }
    }
}

/**
 * Lemma: the augmentation returned by consistent_step_commute_aug4_choice
 * does make (config1, aug1) -> (config1.step(op2), aug4) -> (config3, aug3)
 * a consistent trace
 */
proof fn lemma_consistent_step_commute_aug4_choice(
    op1: OperatorIndex, op2: OperatorIndex,
    config1: Configuration, aug1: PermissionAugmentation,
    config2: Configuration, aug2: PermissionAugmentation,
    config3: Configuration, aug3: PermissionAugmentation,
)
    requires
        config1.valid(),
        config2.valid(),
        config3.valid(),
        config1.fireable(op1),
        config1.fireable(op2),
        op1 != op2,

        // config1 -> config2 -> config3 is a consistent trace
        consistent_step(op1, config1, aug1, config2, aug2),
        consistent_step(op2, config2, aug2, config3, aug3),

        // This is left as an assumption
        config1.step(op2).step(op1) == config3,

    ensures
        ({
            let config4 = config1.step(op2);
            let aug4 = consistent_step_commute_aug4_choice(op1, op2, config1, aug1, config2, aug2, config3, aug3);
            
            consistent_step(op2, config1, aug1, config4, aug4) &&
            consistent_step(op1, config4, aug4, config3, aug3)
        }),
{
    let config4 = config1.step(op2);
    let aug4 = consistent_step_commute_aug4_choice(op1, op2, config1, aug1, config2, aug2, config3, aug3);
    
    config1.lemma_step_valid(op2);
    config1.lemma_step_independence(op1, op2);
    config4.lemma_step_valid(op1);
    config2.lemma_step_valid(op2);
    
    assert(aug4.valid(config4)) by {
        assert(aug4.aug_map.dom() =~= config4.graph.channels);
        
        assert forall |channel: ChannelIndex|
            #[trigger] config4.is_channel(channel)
        implies
            aug4.aug_map[channel].len() == config4.channels[channel].len() &&
            (forall |i: int| 0 <= i < aug4.aug_map[channel].len() ==> (#[trigger] aug4.aug_map[channel][i]).valid()) by
        {
            reveal(Configuration::step);
        }

        assert forall |channel1: ChannelIndex, channel2: ChannelIndex, i: int, j: int|
            config4.is_channel(channel1) && config4.is_channel(channel2) &&
            0 <= i < aug4.aug_map[channel1].len() && 0 <= j < aug4.aug_map[channel2].len() &&
            (channel1 != channel2 || i != j)
        implies
            aug4.aug_map[channel1][i].disjoint(aug4.aug_map[channel2][j]) by
        {
            // TODO: this should be true
            assume(false);
        }
    }

    assert(consistent_step(op2, config1, aug1, config4, aug4)) by {
        let inputs = config1.get_op_input_channels(op2);
        let outputs = config1.get_op_output_channels(op2);

        let input_perms = Seq::new(inputs.len(), |i: int| aug1.aug_map[inputs[i]].first());
        let output_perms = Seq::new(outputs.len(), |i: int| aug4.aug_map[outputs[i]].last());

        let input_perms_after_op1 = Seq::new(inputs.len(), |i: int| aug2.aug_map[inputs[i]].first());
        let output_perms_after_op1_op2 = Seq::new(outputs.len(), |i: int| aug3.aug_map[outputs[i]].last());

        // config1.lemma_step_independence(op1, op2);
        assert(input_perms =~= input_perms_after_op1);
        assert(output_perms =~= output_perms_after_op1_op2);

        // assert forall |channel: ChannelIndex|
        //     #[trigger] config1.is_channel(channel)
        // implies {
        //     if !inputs.contains(channel) && !outputs.contains(channel) {
        //         aug1.aug_map[channel] =~= aug4.aug_map[channel]
        //     } else if inputs.contains(channel) && !outputs.contains(channel) {
        //         aug4.aug_map[channel] =~= aug1.aug_map[channel].drop_first()
        //     } else if !inputs.contains(channel) && outputs.contains(channel) {
        //         aug4.aug_map[channel].drop_last() =~= aug1.aug_map[channel]
        //     } else {
        //         aug4.aug_map[channel].drop_last() =~= aug1.aug_map[channel].drop_first()
        //     }
        // } by
        // {
        //     if !inputs.contains(channel) && !outputs.contains(channel) {
        //         // assert(aug1.aug_map[channel] =~= aug4.aug_map[channel]);
        //     } else if inputs.contains(channel) && !outputs.contains(channel) {
        //         // assert(aug4.aug_map[channel] =~= aug1.aug_map[channel].drop_first());
        //     } else if !inputs.contains(channel) && outputs.contains(channel) {
        //         // assert(aug4.aug_map[channel].drop_last() =~= aug1.aug_map[channel]);
        //     } else {
        //         // assert(aug4.aug_map[channel].drop_last() =~= aug1.aug_map[channel].drop_first());   
        //     }
        // }

        // assert(Permission::union_of(input_perms).contains(Permission::union_of(output_perms)));

        // assert(config1.operators[op2].is_Read() ==>
        //     Permission::union_of(input_perms).has_read(config1.get_op_input_values(op2)[0].as_address()));
        
        // assert(config1.operators[op2].is_Write() ==>
        //     Permission::union_of(input_perms).has_write(config1.get_op_input_values(op2)[0].as_address()));

    }

    assert(consistent_step(op1, config4, aug4, config3, aug3)) by {
        let op1_inputs = config4.get_op_input_channels(op1);
        let op1_outputs = config4.get_op_output_channels(op1);
        
        let op2_inputs = config1.get_op_input_channels(op2);
        let op2_outputs = config1.get_op_output_channels(op2);

        assert(op1_inputs == config1.get_op_input_channels(op1));
        assert(op1_outputs == config1.get_op_output_channels(op1));

        assert(op2_inputs == config2.get_op_input_channels(op2));
        assert(op2_outputs == config2.get_op_output_channels(op2));

        assert(
            config4.valid() &&
            config3.valid() &&
            config3 == config4.step(op1) &&
            aug4.valid(config4) &&
            aug3.valid(config3)
        );

        assert forall |channel: ChannelIndex|
            #[trigger] config4.is_channel(channel)
        implies {
            if !op1_inputs.contains(channel) && !op1_outputs.contains(channel) {
                aug3.aug_map[channel] =~= aug4.aug_map[channel]
            } else if op1_inputs.contains(channel) && !op1_outputs.contains(channel) {
                aug3.aug_map[channel] =~= aug4.aug_map[channel].drop_first()
            } else if !op1_inputs.contains(channel) && op1_outputs.contains(channel) {
                aug3.aug_map[channel].drop_last() =~= aug4.aug_map[channel]
            } else {
                aug3.aug_map[channel].drop_last() =~= aug4.aug_map[channel].drop_first()
            }
        } by
        {
            reveal(Configuration::step);

            if (!op2_inputs.contains(channel) && op2_outputs.contains(channel) &&
                op1_inputs.contains(channel) && !op1_outputs.contains(channel)) {

                // Let A be some sequence
                // aug1 = [a] + A
                // aug2 = A
                // aug3 = A + [b]
                // aug4 = [a] + A + [b]

                assert(aug1.aug_map[channel].len() >= 1);
                assert(aug3.aug_map[channel].len() >= 1);

                let a = aug1.aug_map[channel].first();
                let b = aug3.aug_map[channel].last();

                assert(aug4.aug_map[channel] =~= seq![a] + aug2.aug_map[channel] + seq![b]);
                assert(aug3.aug_map[channel] =~= aug2.aug_map[channel] + seq![b]);
            }
        }

        let input_perms_after_op2 = Seq::new(op1_inputs.len(), |i: int| aug4.aug_map[op1_inputs[i]].first());
        let output_perms_after_op2_op1 = Seq::new(op1_outputs.len(), |i: int| aug3.aug_map[op1_outputs[i]].last());

        let input_perms_init = Seq::new(op1_inputs.len(), |i: int| aug1.aug_map[op1_inputs[i]].first());
        let output_perms_after_op1 = Seq::new(op1_outputs.len(), |i: int| aug2.aug_map[op1_outputs[i]].last());

        assert(input_perms_after_op2 =~= input_perms_init);
        assert(output_perms_after_op2_op1 =~= output_perms_after_op1);
    }
}

/**
 * Lemma: If we have two consistent steps with both operators fireable in the initial config,
 * then their order of execution can be swapped without changing the result.
 */
proof fn lemma_consistent_step_commute(
    op1: OperatorIndex, op2: OperatorIndex,
    config1: Configuration, aug1: PermissionAugmentation,
    config2: Configuration, aug2: PermissionAugmentation,
    config3: Configuration, aug3: PermissionAugmentation,
)
    requires
        config1.valid(),
        config2.valid(),
        config3.valid(),
        config1.fireable(op1),
        config1.fireable(op2),
        op1 != op2,

        // config1 -> config2 -> config3 is a consistent trace
        consistent_step(op1, config1, aug1, config2, aug2),
        consistent_step(op2, config2, aug2, config3, aug3),

    ensures
        config1.step(op2).step(op1) == config3,

        // (config1, aug1) -> (config1.step(op2), aug4) -> (config3, aug3)
        // is a consistent trace
        ({
            let config4 = config1.step(op2);
            let aug4 = consistent_step_commute_aug4_choice(op1, op2, config1, aug1, config2, aug2, config3, aug3);
            consistent_step(op2, config1, aug1, config4, aug4) &&
            consistent_step(op1, config4, aug4, config3, aug3)
        }),
{
    assert(config1.step(op2).step(op1) == config3) by {
        if (config1.operators[op1].is_NonMemory() ||
            config1.operators[op2].is_NonMemory() ||
            (config1.operators[op1].is_Read() && config1.operators[op2].is_Read())) {
            config1.lemma_step_non_memory_commute(op1, op2);
        } else {
            // op1 and op2 are accessing different memory locations
            lemma_consistent_step_disjoint_memory_address(op1, op2, config1, aug1, config2, aug2, config3, aug3);

            assert(config3 == config1.step(op2).step(op1)) by {
                reveal(Configuration::step);
                assert(config3.operators =~= config1.step(op2).step(op1).operators);
                assert(config3.memory =~= config1.step(op2).step(op1).memory);
                assert(config3.channels =~~= config1.step(op2).step(op1).channels);
            }
        }
    }

    lemma_consistent_step_commute_aug4_choice(op1, op2, config1, aug1, config2, aug2, config3, aug3);
}

} // verus!
