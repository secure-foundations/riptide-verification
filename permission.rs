use vstd::prelude::*;
use crate::semantics::*;

verus! {

pub struct Permission {
    pub access: Map<Address, Seq<bool>>,
}

pub struct AugmentedConfiguration {
    pub config: Configuration,
    pub aug: Map<ChannelIndex, Seq<Permission>>,
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
    #[verifier(opaque)]
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
    #[verifier(opaque)]
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
    #[verifier(opaque)]
    pub open spec fn has_read(self, addr: Address) -> bool
        recommends self.valid()
    {
        exists |i: int| 0 <= i < permission_write_split() && self.access[addr][i]
    }

    /**
     * Whether the permission has write permission on an address
     */
    #[verifier(opaque)]
    pub open spec fn has_write(self, addr: Address) -> bool
        recommends self.valid()
    {
        forall |i: int| 0 <= i < permission_write_split() ==> self.access[addr][i]
    }

    /**
     * Condition for self to contain other
     */
    #[verifier(opaque)]
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

    // Some auto facts about permissions, union, disjoint, and contains
    pub proof fn lemma_union_disjoint_reasoning_auto()
        ensures
            (forall |perms: Seq<Permission>, i: int| 0 <= i < perms.len() ==> Self::union_of(perms).contains(#[trigger] perms[i])),
            (forall |perms: Seq<Permission>, other: Permission, i: int|
                0 <= i < perms.len() &&
                Self::union_of(perms).disjoint(other)
                ==> #[trigger] perms[i].disjoint(other)),
            (forall |perm1: Permission, perm2: Permission, perm3: Permission|
                #![trigger perm2.contains(perm1), perm2.disjoint(perm3)]
                #![trigger perm2.contains(perm1), perm1.disjoint(perm3)]
                #![trigger perm2.disjoint(perm3), perm1.disjoint(perm3)]
                perm2.contains(perm1) && perm2.disjoint(perm3) ==>
                perm1.disjoint(perm3)),
            (forall |perm1: Permission, perm2: Permission|
                #![trigger perm1.disjoint(perm2)]
                #![trigger perm2.disjoint(perm1)]
                perm1.disjoint(perm2) == perm2.disjoint(perm1)),
            (forall |perm1: Permission, perm2: Permission, perm3: Permission|
                perm1.contains(perm2) && perm2.contains(perm3) ==>
                perm1.contains(perm3)),
    {
        reveal(Permission::union_of);
        reveal(Permission::contains);
        reveal(Permission::disjoint);
    }

    /**
     * Lemma: Let perms1 and perms2 be two lists of permissions.
     * If any permission in perms1 is disjoint from any permission
     * in perms2, then the union of perms1 is disjoint from the
     * union of perms2.
     */
    pub proof fn lemma_mutually_disjoint_union(perms1: Seq<Permission>, perms2: Seq<Permission>)
        requires
            perms1.len() > 0,
            perms2.len() > 0,

            forall |i: int, j: int|
                #![trigger perms1[i], perms2[j]]
                0 <= i < perms1.len() && 0 <= j < perms2.len() ==>
                perms1[i].disjoint(perms2[j])

        ensures
            Permission::union_of(perms1).disjoint(Permission::union_of(perms2))
    {
        reveal(Permission::union_of);
        reveal(Permission::disjoint);
    }
}

impl AugmentedConfiguration {
    /**
     * Whether an augmented configuration is valid
     */
    pub open spec fn valid(self) -> bool
    {
        self.config.valid() &&

        // Domain should be exactly the set of channels
        self.aug.dom() =~= self.config.graph.channels &&

        // Each permission should be valid
        (forall |channel: ChannelIndex|
            #[trigger] self.config.is_channel(channel) ==>
            self.aug[channel].len() == self.config.channels[channel].len() &&
            forall |i: int| 0 <= i < self.aug[channel].len() ==> (#[trigger] self.aug[channel][i]).valid()) &&

        // Permissions should be mutually disjoint
        (forall |channel1: ChannelIndex, channel2: ChannelIndex, i: int, j: int|
            #![trigger self.aug[channel1][i], self.aug[channel2][j]]
            self.config.is_channel(channel1) && self.config.is_channel(channel2) &&
            0 <= i < self.aug[channel1].len() && 0 <= j < self.aug[channel2].len() &&
            (channel1 != channel2 || i != j) ==>
            self.aug[channel1][i].disjoint(self.aug[channel2][j]))
    }

    pub open spec fn get_op_input_permissions(self, op: OperatorIndex) -> Seq<Permission>
        recommends self.valid()
    {
        let inputs = self.config.get_op_input_channels(op);
        Seq::new(inputs.len(), |i: int| self.aug[inputs[i]].first())
    }
}

/**
 * Defines what it means for an augmented transition/step
 * aug_config1 -> aug_config2
 * to be consistent,
 */
pub open spec fn consistent_step(
    op: OperatorIndex,
    aug_config1: AugmentedConfiguration,
    aug_config2: AugmentedConfiguration,
) -> bool
{
    let inputs = aug_config1.config.get_op_input_channels(op);
    let outputs = aug_config1.config.get_op_output_channels(op);

    let input_perms = Seq::new(inputs.len(), |i: int| aug_config1.aug[inputs[i]].first());
    let output_perms = Seq::new(outputs.len(), |i: int| aug_config2.aug[outputs[i]].last());

    aug_config1.valid() &&
    aug_config2.valid() &&
    aug_config2.config == aug_config1.config.step(op) &&

    // Four cases:
    // 1. channel not in inputs or outputs
    // 2. channel in inputs only
    // 3. channel in outputs only
    // 4. channel in both inputs and outputs
    (forall |channel: ChannelIndex|
        #[trigger] aug_config1.config.is_channel(channel) ==> {
            if !inputs.contains(channel) && !outputs.contains(channel) {
                // The permissions are unchanged in case 1
                aug_config2.aug[channel] =~= aug_config1.aug[channel]
            } else if inputs.contains(channel) && !outputs.contains(channel) {
                // Case 2
                aug_config2.aug[channel] =~= aug_config1.aug[channel].drop_first()
            } else if !inputs.contains(channel) && outputs.contains(channel) {
                // Case 3
                aug_config2.aug[channel].drop_last() =~= aug_config1.aug[channel]
            } else {
                // Case 4
                aug_config2.aug[channel].drop_last() =~= aug_config1.aug[channel].drop_first()
            }
        }
    ) &&

    // Union of input perms is less than equal to the union of output perms
    Permission::union_of(input_perms).contains(Permission::union_of(output_perms)) &&

    // If the operator is a read/write, we require suitable permissions
    (aug_config1.config.operators[op].is_Read() ==>
        Permission::union_of(input_perms).has_read(aug_config1.config.get_op_input_values(op)[0].as_address())) &&
        
    (aug_config1.config.operators[op].is_Write() ==>
        Permission::union_of(input_perms).has_write(aug_config1.config.get_op_input_values(op)[0].as_address()))
}

/**
 * Lemma: Input permissions are not changed after firing
 * a different operator (similar to lemma_step_independence).
 */
pub proof fn lemma_step_independent_input_permissions(
    op1: OperatorIndex, op2: OperatorIndex,
    aug_config1: AugmentedConfiguration,
    aug_config2: AugmentedConfiguration,
)
    requires
        aug_config1.config.fireable(op1),
        aug_config1.config.fireable(op2),
        op1 != op2,
        consistent_step(op1, aug_config1, aug_config2),

    ensures
        aug_config1.get_op_input_permissions(op2) == aug_config2.get_op_input_permissions(op2)
{
    aug_config1.config.lemma_step_independence(op1, op2);
    assert(aug_config1.get_op_input_permissions(op2) =~= aug_config2.get_op_input_permissions(op2));
}

/**
 * Lemma: If both op1 and op2 are fireable in an augmented
 * configuration, then their input permissions must be
 * disjoint (by the augmentation being valid).
 */
proof fn lemma_fireable_disjoint_input_permissions(
    op1: OperatorIndex, op2: OperatorIndex,
    aug_config: AugmentedConfiguration,
)
    requires
        aug_config.valid(),
        aug_config.config.fireable(op1),
        aug_config.config.fireable(op2),
        op1 != op2,

    ensures
        Permission::union_of(aug_config.get_op_input_permissions(op1)).disjoint(Permission::union_of(aug_config.get_op_input_permissions(op2))),
{
    let op1_input_perms = aug_config.get_op_input_permissions(op1);
    let op2_input_perms = aug_config.get_op_input_permissions(op2);

    assert(Permission::union_of(op1_input_perms).disjoint(Permission::union_of(op2_input_perms))) by {
        if op1_input_perms.len() > 0 && op2_input_perms.len() > 0 {
            Permission::lemma_mutually_disjoint_union(op1_input_perms, op2_input_perms);
            Permission::lemma_union_disjoint_reasoning_auto();
        } else {
            reveal(Permission::union_of);
            reveal(Permission::disjoint);
        }
    }
}

/**
 * Lemma: If both **memory** operators op1 and op2 are fireable in a
 * configuration (and one of op1 and op2 is a write), and op1 and op2
 * can be fired in consistent steps, then their accessed memory addresses
 * must be different.
 */
proof fn lemma_consistent_steps_have_distinct_memory_addresses(
    op1: OperatorIndex, op2: OperatorIndex,
    aug_config1: AugmentedConfiguration,
    aug_config2: AugmentedConfiguration,
    aug_config3: AugmentedConfiguration,
)
    requires
        aug_config1.valid(),
        aug_config2.valid(),
        aug_config3.valid(),
        aug_config1.config.fireable(op1),
        aug_config1.config.fireable(op2),
        op1 != op2,

        consistent_step(op1, aug_config1, aug_config2),
        consistent_step(op2, aug_config2, aug_config3),

        // op1 and op2 are memory operators, and at least one of op1 and op2 is a write
        aug_config1.config.operators[op1].is_Read() || aug_config1.config.operators[op1].is_Write(),
        aug_config1.config.operators[op2].is_Read() || aug_config1.config.operators[op2].is_Write(),
        aug_config1.config.operators[op1].is_Write() || aug_config1.config.operators[op2].is_Write(),

    ensures
        aug_config1.config.get_op_input_values(op1)[0].as_address() != aug_config1.config.get_op_input_values(op2)[0].as_address()
{
    let op1_inputs = aug_config1.config.get_op_input_channels(op1);
    let op2_inputs = aug_config1.config.get_op_input_channels(op2);

    let op1_address = aug_config1.config.get_op_input_values(op1)[0].as_address();
    let op2_address = aug_config1.config.get_op_input_values(op2)[0].as_address();

    let op1_input_perms_init = aug_config1.get_op_input_permissions(op1);
    let op2_input_perms_init = aug_config1.get_op_input_permissions(op2);
    // let op2_input_perms_after_op1 = aug_config2.get_op_input_permissions(op2);

    aug_config1.config.lemma_step_independence(op1, op2);
    lemma_step_independent_input_permissions(op1, op2, aug_config1, aug_config2);
    lemma_fireable_disjoint_input_permissions(op1, op2, aug_config1);
    // assert(op2_input_perms_init =~= op2_input_perms_after_op1);

    assert(op1_address != op2_address) by {
        let op1_input_perm_union = Permission::union_of(op1_input_perms_init);
        let op2_input_perm_union = Permission::union_of(op2_input_perms_init);

        // Proof by contradiction
        if op1_address == op2_address {
            if (aug_config1.config.operators[op1].is_Write() && aug_config1.config.operators[op2].is_Write()) {
                let trigger = op1_input_perm_union.access[op1_address][0];
            }
            reveal(Permission::union_of);
            reveal(Permission::disjoint);
            reveal(Permission::has_read);
            reveal(Permission::has_write);
            assert(false);
        }
    }
}

/**
 * Constructs a valid augmented configuration aug_config4 for config1.step(op2)
 * such that (config1, aug1) -> (config1.step(op2), aug4) -> (config3, aug3)
 * is a consistent trace.
 */
spec fn consistent_steps_commute_augmentation(
    op1: OperatorIndex, op2: OperatorIndex,
    aug_config1: AugmentedConfiguration,
    aug_config2: AugmentedConfiguration,
    aug_config3: AugmentedConfiguration,
) -> AugmentedConfiguration
{
    let inputs = aug_config1.config.get_op_input_channels(op2);
    let outputs = aug_config1.config.get_op_output_channels(op2);

    let input_perms = aug_config1.get_op_input_permissions(op2);
    let output_perms = Seq::new(outputs.len(), |i: int| aug_config3.aug[outputs[i]].last());

    let new_aug = Map::new(
        |channel: ChannelIndex| aug_config1.config.is_channel(channel),
        |channel: ChannelIndex|
            if !inputs.contains(channel) && !outputs.contains(channel) {
                aug_config1.aug[channel]
            } else if inputs.contains(channel) && !outputs.contains(channel) {
                aug_config1.aug[channel].drop_first()
            } else if !inputs.contains(channel) && outputs.contains(channel) {
                let output_index = outputs.index_of(channel);
                aug_config1.aug[channel].push(output_perms[output_index])
            } else {
                let output_index = outputs.index_of(channel);
                aug_config1.aug[channel].drop_first().push(output_perms[output_index])
            }
    );

    AugmentedConfiguration { config: aug_config1.config.step(op2), aug: new_aug }
}

/**
 * (Some technical fact)
 * Lemma: If (channel1, i) is the position of an output permission of op2 (in aug4)
 * and (channel2, j) is another position of a permission in aug4,
 * then these two permissions are disjoint
 */
proof fn lemma_consistent_steps_commute_augmentation_helper(
    op1: OperatorIndex, op2: OperatorIndex,
    aug_config1: AugmentedConfiguration,
    aug_config2: AugmentedConfiguration,
    aug_config3: AugmentedConfiguration,

    channel1: ChannelIndex, i: int,
    channel2: ChannelIndex, j: int,
)
    requires
        aug_config1.valid(),
        aug_config2.valid(),
        aug_config3.valid(),
        aug_config1.config.fireable(op1),
        aug_config1.config.fireable(op2),
        op1 != op2,

        consistent_step(op1, aug_config1, aug_config2),
        consistent_step(op2, aug_config2, aug_config3),

        aug_config1.config.step(op2).step(op1) == aug_config3.config,

        ({
            let aug_config4 = consistent_steps_commute_augmentation(op1, op2, aug_config1, aug_config2, aug_config3);

            aug_config4.config.is_channel(channel1) &&
            aug_config4.config.is_channel(channel2) &&
            
            0 <= i < aug_config4.aug[channel1].len() &&
            0 <= j < aug_config4.aug[channel2].len() &&
            (channel1 != channel2 || i != j) &&

            // (channel1, i) is the position of an output permission of op2
            aug_config1.config.get_op_output_channels(op2).contains(channel1) &&
            i == aug_config4.aug[channel1].len() - 1
        }),

    ensures
        ({
            let aug_config4 = consistent_steps_commute_augmentation(op1, op2, aug_config1, aug_config2, aug_config3);
            aug_config4.aug[channel1][i].disjoint(aug_config4.aug[channel2][j])
        }),
{
    let aug_config4 = consistent_steps_commute_augmentation(op1, op2, aug_config1, aug_config2, aug_config3);

    let op1_inputs = aug_config1.config.get_op_input_channels(op1);
    let op1_outputs = aug_config1.config.get_op_output_channels(op1);

    let op2_inputs = aug_config1.config.get_op_input_channels(op2);
    let op2_outputs = aug_config1.config.get_op_output_channels(op2);

    aug_config1.config.lemma_step_independence(op1, op2);

    assert(aug_config4.aug[channel1][i] == aug_config3.aug[channel1].last());

    if op1_inputs.contains(channel2) && j == 0 {
        // In this case, channel2 is an input channel of op1 and j is exactly the first input
        // to be consumed.
        // i.e. aug_config4.aug[channel2][j] <= op1 input permissions
        // aug_config4.aug[channel1][i] <= op2 output permissions
        // Since disjoint(op1 input permissions, op2 input permissions) (by lemma_consistent_steps_have_distinct_memory_addresses)
        // and op2 output permissions <= op2 input permissions
        // we have disjoint(aug_config4.aug[channel1][i], aug_config4.aug[channel2][j])

        let op1_input_perms_init = aug_config1.get_op_input_permissions(op1);
        let op2_input_perms_init = aug_config1.get_op_input_permissions(op2);

        let op2_input_perms_after_op1 = aug_config2.get_op_input_permissions(op2);
        let op2_output_perms_after_op1 = Seq::new(op2_outputs.len(), |i: int| aug_config3.aug[op2_outputs[i]].last());

        Permission::lemma_union_disjoint_reasoning_auto();

        assert(Permission::union_of(op2_input_perms_init).contains(aug_config4.aug[channel1][i])) by {
            assert(op2_output_perms_after_op1[op2_outputs.index_of(channel1)] == aug_config4.aug[channel1][i]);
            assert(Permission::union_of(op2_output_perms_after_op1).contains(aug_config4.aug[channel1][i]));

            // By validity of the transition aug_config2 -> aug_config3
            assert(Permission::union_of(op2_input_perms_after_op1).contains(Permission::union_of(op2_output_perms_after_op1)));
            assert(op2_input_perms_init =~= op2_input_perms_after_op1) by {
                lemma_step_independent_input_permissions(op1, op2, aug_config1, aug_config2);
            }
            assert(Permission::union_of(op2_input_perms_init).contains(Permission::union_of(op2_output_perms_after_op1)));
        }

        assert(Permission::union_of(op1_input_perms_init).contains(aug_config4.aug[channel2][j])) by {
            assert(op1_input_perms_init[op1_inputs.index_of(channel2)] == aug_config4.aug[channel2][j]);
        }

        assert(Permission::union_of(op1_input_perms_init).disjoint(Permission::union_of(op2_input_perms_init))) by {
            lemma_fireable_disjoint_input_permissions(op1, op2, aug_config1);
        }

        assert(aug_config4.aug[channel1][i].disjoint(aug_config4.aug[channel2][j]));
    } else {
        // Some terms to trigger QIs
        // Basically a more detailed proof has to do case analysis
        // on the position of (channel2, j) (e.g. whether it's an
        // input/output of op1/op2, and how their positions change
        // as a result)
        let _ = aug_config3.aug[channel2][j];
        let _ = aug_config3.aug[channel2][j - 1];
        let _ = aug_config2.aug[channel2][j + 1];
        let _ = aug_config2.aug[channel2][j];
        let _ = aug_config2.aug[channel2][j - 1];
        let _ = aug_config1.aug[channel2].drop_first()[j - 1];

        assert(forall |seq: Seq<Permission>, i: int| 0 <= i < seq.len() - 1 ==>
            #[trigger] seq.drop_first()[i] == seq[i + 1]);

        assert(aug_config4.aug[channel1][i].disjoint(aug_config4.aug[channel2][j])) by {
            // TODO: try reduce this to some generic facts about Configuration::step
            reveal(Configuration::step);
        }
    }
}

/**
 * Lemma: the augmentation returned by consistent_steps_commute_augmentation
 * does make (config1, aug1) -> (config1.step(op2), aug4) -> (config3, aug3)
 * a consistent trace
 */
proof fn lemma_consistent_steps_commute_augmentation(
    op1: OperatorIndex, op2: OperatorIndex,
    aug_config1: AugmentedConfiguration,
    aug_config2: AugmentedConfiguration,
    aug_config3: AugmentedConfiguration,
)
    requires
        aug_config1.valid(),
        aug_config2.valid(),
        aug_config3.valid(),
        aug_config1.config.fireable(op1),
        aug_config1.config.fireable(op2),
        op1 != op2,

        consistent_step(op1, aug_config1, aug_config2),
        consistent_step(op2, aug_config2, aug_config3),

        aug_config1.config.step(op2).step(op1) == aug_config3.config,

    ensures
        ({
            let aug_config4 = consistent_steps_commute_augmentation(op1, op2, aug_config1, aug_config2, aug_config3);
            consistent_step(op2, aug_config1, aug_config4) &&
            consistent_step(op1, aug_config4, aug_config3)
        }),
{
    let aug_config4 = consistent_steps_commute_augmentation(op1, op2, aug_config1, aug_config2, aug_config3);
    
    aug_config1.config.lemma_step_valid(op2);
    
    assert(aug_config4.valid()) by {
        let op1_inputs = aug_config1.config.get_op_input_channels(op1);
        let op1_outputs = aug_config1.config.get_op_output_channels(op1);

        let op2_inputs = aug_config1.config.get_op_input_channels(op2);
        let op2_outputs = aug_config1.config.get_op_output_channels(op2);
        
        assert forall |channel: ChannelIndex|
            #[trigger] aug_config4.config.is_channel(channel)
        implies
            aug_config4.aug[channel].len() == aug_config4.config.channels[channel].len() &&
            (forall |i: int| 0 <= i < aug_config4.aug[channel].len() ==> (#[trigger] aug_config4.aug[channel][i]).valid()) by
        {
            reveal(Configuration::step);
        }

        assert forall |channel1: ChannelIndex, channel2: ChannelIndex, i: int, j: int|
            aug_config4.config.is_channel(channel1) && aug_config4.config.is_channel(channel2) &&
            0 <= i < aug_config4.aug[channel1].len() && 0 <= j < aug_config4.aug[channel2].len() &&
            (channel1 != channel2 || i != j)
        implies
            aug_config4.aug[channel1][i].disjoint(aug_config4.aug[channel2][j]) by
        {
            if op2_outputs.contains(channel1) && i == aug_config4.aug[channel1].len() - 1 {
                lemma_consistent_steps_commute_augmentation_helper(op1, op2, aug_config1, aug_config2, aug_config3, channel1, i, channel2, j);
            } else if op2_outputs.contains(channel2) && j == aug_config4.aug[channel2].len() - 1 {
                lemma_consistent_steps_commute_augmentation_helper(op1, op2, aug_config1, aug_config2, aug_config3, channel2, j, channel1, i);
                Permission::lemma_union_disjoint_reasoning_auto();
            }
        }
    }

    assert(consistent_step(op2, aug_config1, aug_config4)) by {
        let inputs = aug_config1.config.get_op_input_channels(op2);
        let outputs = aug_config1.config.get_op_output_channels(op2);

        let input_perms = Seq::new(inputs.len(), |i: int| aug_config1.aug[inputs[i]].first());
        let output_perms = Seq::new(outputs.len(), |i: int| aug_config4.aug[outputs[i]].last());

        let input_perms_after_op1 = Seq::new(inputs.len(), |i: int| aug_config2.aug[inputs[i]].first());
        let output_perms_after_op1_op2 = Seq::new(outputs.len(), |i: int| aug_config3.aug[outputs[i]].last());

        aug_config1.config.lemma_step_independence(op1, op2);
        assert(input_perms =~= input_perms_after_op1);
        assert(output_perms =~= output_perms_after_op1_op2);
    }

    assert(consistent_step(op1, aug_config4, aug_config3)) by {
        let op1_inputs = aug_config4.config.get_op_input_channels(op1);
        let op1_outputs = aug_config4.config.get_op_output_channels(op1);
        
        let op2_inputs = aug_config1.config.get_op_input_channels(op2);
        let op2_outputs = aug_config1.config.get_op_output_channels(op2);

        aug_config1.config.lemma_step_independence(op1, op2);

        assert(op1_inputs == aug_config1.config.get_op_input_channels(op1));
        assert(op1_outputs == aug_config1.config.get_op_output_channels(op1));

        assert(op2_inputs == aug_config2.config.get_op_input_channels(op2));
        assert(op2_outputs == aug_config2.config.get_op_output_channels(op2));

        assert forall |channel: ChannelIndex|
            #[trigger] aug_config4.config.is_channel(channel)
        implies {
            if !op1_inputs.contains(channel) && !op1_outputs.contains(channel) {
                aug_config3.aug[channel] =~= aug_config4.aug[channel]
            } else if op1_inputs.contains(channel) && !op1_outputs.contains(channel) {
                aug_config3.aug[channel] =~= aug_config4.aug[channel].drop_first()
            } else if !op1_inputs.contains(channel) && op1_outputs.contains(channel) {
                aug_config3.aug[channel].drop_last() =~= aug_config4.aug[channel]
            } else {
                aug_config3.aug[channel].drop_last() =~= aug_config4.aug[channel].drop_first()
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

                assert(aug_config1.aug[channel].len() >= 1);
                assert(aug_config3.aug[channel].len() >= 1);

                let a = aug_config1.aug[channel].first();
                let b = aug_config3.aug[channel].last();

                assert(aug_config4.aug[channel] =~= seq![a] + aug_config2.aug[channel] + seq![b]);
                assert(aug_config3.aug[channel] =~= aug_config2.aug[channel] + seq![b]);
            }
        }

        let input_perms_after_op2 = Seq::new(op1_inputs.len(), |i: int| aug_config4.aug[op1_inputs[i]].first());
        let output_perms_after_op2_op1 = Seq::new(op1_outputs.len(), |i: int| aug_config3.aug[op1_outputs[i]].last());

        let input_perms_init = Seq::new(op1_inputs.len(), |i: int| aug_config1.aug[op1_inputs[i]].first());
        let output_perms_after_op1 = Seq::new(op1_outputs.len(), |i: int| aug_config2.aug[op1_outputs[i]].last());

        assert(input_perms_after_op2 =~= input_perms_init);
        assert(output_perms_after_op2_op1 =~= output_perms_after_op1) by {
            aug_config4.config.lemma_step_valid(op1);
            aug_config2.config.lemma_step_valid(op2);
        }
    }
}

/**
 * Lemma: If we have two consistent steps with both operators fireable in the initial config,
 * then their order of execution can be swapped without changing the result.
 */
proof fn lemma_consistent_steps_commute(
    op1: OperatorIndex, op2: OperatorIndex,
    aug_config1: AugmentedConfiguration,
    aug_config2: AugmentedConfiguration,
    aug_config3: AugmentedConfiguration,
)
    requires
        aug_config1.valid(),
        aug_config2.valid(),
        aug_config3.valid(),
        aug_config1.config.fireable(op1),
        aug_config1.config.fireable(op2),
        op1 != op2,

        // aug_config1 -> aug_config2 -> aug_config3 is a consistent trace
        consistent_step(op1, aug_config1, aug_config2),
        consistent_step(op2, aug_config2, aug_config3),

    ensures
        aug_config1.config.step(op2).step(op1) == aug_config3.config,

        // aug_config1 -> aug_config4 -> aug_config3 is a consistent trace
        ({
            let aug_config4 = consistent_steps_commute_augmentation(op1, op2, aug_config1, aug_config2, aug_config3);
            consistent_step(op2, aug_config1, aug_config4) &&
            consistent_step(op1, aug_config4, aug_config3)
        }),
{
    assert(aug_config1.config.step(op2).step(op1) == aug_config3.config) by {
        if (aug_config1.config.operators[op1].is_NonMemory() ||
            aug_config1.config.operators[op2].is_NonMemory() ||
            (aug_config1.config.operators[op1].is_Read() && aug_config1.config.operators[op2].is_Read())) {
            aug_config1.config.lemma_step_non_memory_commute(op1, op2);
        } else {
            // op1 and op2 are accessing different memory locations
            lemma_consistent_steps_have_distinct_memory_addresses(op1, op2, aug_config1, aug_config2, aug_config3);

            assert(aug_config3.config == aug_config1.config.step(op2).step(op1)) by {
                reveal(Configuration::step);
                assert(aug_config3.config.operators =~= aug_config1.config.step(op2).step(op1).operators);
                assert(aug_config3.config.memory =~= aug_config1.config.step(op2).step(op1).memory);
                assert(aug_config3.config.channels =~~= aug_config1.config.step(op2).step(op1).channels);
            }
        }
    }

    lemma_consistent_steps_commute_augmentation(op1, op2, aug_config1, aug_config2, aug_config3);
}

} // verus!
