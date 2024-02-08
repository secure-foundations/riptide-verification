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

pub struct AugmentedTrace {
    pub configs: Seq<AugmentedConfiguration>,
    pub operators: Seq<OperatorIndex>,
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

impl AugmentedTrace {
    /**
     * Consistent trace. We allow 0 length trace (which is just a single consistent configuration)
     */
    pub open spec fn valid(self) -> bool
    {
        self.operators.len() + 1 == self.configs.len() &&
        self.configs.first().valid() && // in case length = 0
        (forall |i: int| 0 <= i < self.operators.len() ==>
            consistent_step(self.operators[i], #[trigger] self.configs[i], self.configs[i + 1]))
    }

    /**
     * Length of the trace = number of operators fired in the trace
     */
    pub open spec fn len(self) -> nat
    {
        self.operators.len() as nat
    }

    pub open spec fn drop_first(self) -> AugmentedTrace
        recommends self.len() > 0
    {
        AugmentedTrace {
            configs: self.configs.drop_first(),
            operators: self.operators.drop_first(),
        }
    }
    
    pub open spec fn drop_last(self) -> AugmentedTrace
        recommends self.len() > 0
    {
        AugmentedTrace {
            configs: self.configs.drop_last(),
            operators: self.operators.drop_last(),
        }
    }

    pub open spec fn drop(self, n: int) -> AugmentedTrace
        recommends self.len() > n
    {
        AugmentedTrace {
            configs: self.configs.take(self.len() - n + 1),
            operators: self.operators.take(self.len() - n),
        }
    }

    /**
     * Skips first n steps
     */
    pub open spec fn skip(self, n: int) -> AugmentedTrace
        recommends self.len() >= n >= 0
    {
        AugmentedTrace {
            configs: self.configs.skip(n),
            operators: self.operators.skip(n),
        }
    }

    /**
     * Takes first n steps
     */
    pub open spec fn take(self, n: int) -> AugmentedTrace
        recommends self.len() >= n >= 0
    {
        AugmentedTrace {
            configs: self.configs.take(n + 1),
            operators: self.operators.take(n),
        }
    }

    /**
     * Add two traces given that self.last is the same as other.first
     */
    pub open spec fn add(self, other: AugmentedTrace) -> AugmentedTrace
        recommends
            other.valid(),
            self.configs.last() == other.configs.first(),
    {
        AugmentedTrace {
            configs: self.configs + other.configs.drop_first(),
            operators: self.operators + other.operators,
        }
    }

    pub open spec fn push(self, op: OperatorIndex, config: AugmentedConfiguration) -> AugmentedTrace
    {
        AugmentedTrace {
            configs: self.configs.push(config),
            operators: self.operators.push(op),
        }
    }

    /**
     * Adds a first step
     */
    pub open spec fn prepend(self, op: OperatorIndex, config: AugmentedConfiguration) -> AugmentedTrace
    {
        AugmentedTrace {
            configs: seq![config] + self.configs,
            operators: seq![op] + self.operators,
        }
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
    aug_config1.config.fireable(op) &&
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

#[verifier(opaque)]
spec fn consistent_trace_commutes(trace: AugmentedTrace) -> AugmentedTrace
    recommends trace.len() > 0
    decreases trace.len()
{
    if trace.len() <= 1 {
        trace
    } else {
        // config_1 -> ... -> config_{n + 1}
        let config_np1 = trace.configs.last();
        let config_n = trace.configs[trace.len() - 1];
        let config_nm1 = trace.configs[trace.len() - 2];

        let op_n = trace.operators.last();
        let op_nm1 = trace.operators[trace.len() - 2];

        // config_nm1 -> config_n -> config_np1 is consistent
        // consistent_step(op_nm1, config_nm1, config_n)
        // consistent_step(op_n, config_n, config_np1)

        let config_n_alt = consistent_steps_commute_augmentation(op_nm1, op_n, config_nm1, config_n, config_np1);
        // consistent_step(op_n, config_nm1, config_n_alt)
        // consistent_step(op_nm1, config_n_alt, config_np1)

        let trace_rest = trace.drop(2).push(op_n, config_n_alt);
        // trace_rest.valid()

        // assert(trace_rest.len() == trace.len() - 1);
        let commuted_trace_rest = consistent_trace_commutes(trace_rest);

        commuted_trace_rest.push(op_nm1, config_np1)
    }
}

/**
 * A helper lemma for lemma_consistent_trace_commutes
 * 
 * Lemma: If the last operator in the trace is firable at the first config,
 * then it is also fireable in the third to last configuration.
 */
proof fn lemma_fireability_propagation(trace: AugmentedTrace)
    requires
        trace.valid(),
        trace.len() > 1,
        trace.configs.first().config.fireable(trace.operators.last()),
        (forall |i: int| 0 <= i < trace.len() - 1 ==> #[trigger] trace.operators[i] != trace.operators.last()),

    ensures
        trace.configs[trace.len() - 2].config.fireable(trace.operators.last()),

    decreases trace.len()
{
    trace.configs.first().config.lemma_step_independence(trace.operators.first(), trace.operators.last());
    assert(trace.configs[1].config.fireable(trace.operators.last()));

    if trace.len() > 2 {
        lemma_fireability_propagation(trace.drop_first());
    }
}

/**
 * Lemma: If we have a consistent trace: aug_config_1 ->^o_1 ... ->^o_n aug_config_{n + 1},
 * firing operators o_1, ..., o_n such that:
 * - o_n is fireable at aug_config_1,
 * - o_n not in { o_1, ..., o_{n - 1} }
 * then we can swap o_n all the way back to the beginning and still get a consistent trace
 * ending in aug_config_{n + 1}
 */
proof fn lemma_consistent_trace_commutes(trace: AugmentedTrace)
    requires
        trace.valid(),
        trace.len() >= 1,
        trace.configs.first().config.fireable(trace.operators.last()),
        (forall |i: int| 0 <= i < trace.len() - 1 ==> trace.operators[i] != trace.operators.last()),

    ensures
        ({
            let commuted_trace = consistent_trace_commutes(trace);

            commuted_trace.valid() &&
            commuted_trace.len() == trace.len() &&
            commuted_trace.configs.first() == trace.configs.first() &&
            commuted_trace.configs.last() == trace.configs.last() &&
            commuted_trace.operators.first() == trace.operators.last() &&
            commuted_trace.operators.drop_first() == trace.operators.drop_last()
        }),

    decreases trace.len()
{
    reveal(consistent_trace_commutes);
    let commuted_trace = consistent_trace_commutes(trace);

    if trace.len() == 1 {
        assert(
            commuted_trace.len() == trace.len() &&
            commuted_trace.configs.first() == trace.configs.first() &&
            commuted_trace.configs.last() == trace.configs.last() &&
            commuted_trace.operators.first() == trace.operators.last() &&
            commuted_trace.operators.drop_first() == trace.operators.drop_last()
        );
    } else {
        let config_np1 = trace.configs.last();
        let config_n = trace.configs[trace.len() - 1];
        let config_nm1 = trace.configs[trace.len() - 2];

        let op_n = trace.operators.last();
        let op_nm1 = trace.operators[trace.len() - 2];

        // config_nm1 -> config_n -> config_np1 is consistent
        assert(consistent_step(op_nm1, config_nm1, config_n));
        assert(consistent_step(op_n, config_n, config_np1));

        let config_n_alt = consistent_steps_commute_augmentation(op_nm1, op_n, config_nm1, config_n, config_np1);
        lemma_fireability_propagation(trace);

        lemma_consistent_steps_commute(op_nm1, op_n, config_nm1, config_n, config_np1);
        assert(consistent_step(op_n, config_nm1, config_n_alt));
        assert(consistent_step(op_nm1, config_n_alt, config_np1));

        let trace_rest = trace.drop(2).push(op_n, config_n_alt);

        let commuted_trace_rest = consistent_trace_commutes(trace_rest);
        lemma_consistent_trace_commutes(trace_rest);

        let commuted_trace = commuted_trace_rest.push(op_nm1, config_np1);

        // assert(commuted_trace_rest.operators.drop_first() == trace_rest.operators.drop_last());
        // assert(commuted_trace.operators =~= commuted_trace_rest.operators.push(op_nm1));
        // assert(commuted_trace.operators.drop_first() =~= commuted_trace_rest.operators.drop_first().push(op_nm1));
        assert(commuted_trace.operators.drop_first() == trace_rest.operators.drop_last().push(op_nm1));
        assert(trace_rest.operators.drop_last().push(op_nm1) =~= trace.operators.drop_last());
    }
}

/**
 * Defines s1 >= s2 as a multiset
 */
#[verifier(opaque)]
spec fn multiset_contains(s1: Seq<OperatorIndex>, s2: Seq<OperatorIndex>) -> bool
{
    s2.to_multiset().subset_of(s1.to_multiset())
}

proof fn lemma_multiset_contains_length(s1: Seq<OperatorIndex>, s2: Seq<OperatorIndex>)
    requires
        multiset_contains(s1, s2),
    
    ensures
        s1.len() >= s2.len(),
{
    reveal(multiset_contains);
    s1.to_multiset_ensures();
    s2.to_multiset_ensures();
    let _ = s1.to_multiset().sub(s2.to_multiset()).len();
}

proof fn lemma_multiset_contains_remove(s1: Seq<OperatorIndex>, s2: Seq<OperatorIndex>, i: int, j: int)
    requires
        multiset_contains(s1, s2),
        0 <= i < s1.len(),
        0 <= j < s2.len(),
        s1[i] == s2[j],
    
    ensures
        multiset_contains(s1.remove(i), s2.remove(j)),
{
    reveal(multiset_contains);
    s1.to_multiset_ensures();
    s2.to_multiset_ensures();

    let _ = s1.to_multiset().count(s1[i]);
    let _ = s2.to_multiset().count(s2[j]);

    let s1_1 = s1.subrange(0, i);
    let s1_2 = seq![s1[i]];
    let s1_3 = s1.skip(i + 1);

    let s2_1 = s2.subrange(0, j);
    let s2_2 = seq![s2[j]];
    let s2_3 = s2.skip(j + 1);

    assert(s1 =~= s1_1 + s1_2 + s1_3);
    assert(s2 =~= s2_1 + s2_2 + s2_3);

    vstd::seq_lib::lemma_multiset_commutative(s1_1 + s1_2, s1_3);
    vstd::seq_lib::lemma_multiset_commutative(s1_1 + s1_3, s1_2);
    vstd::seq_lib::lemma_multiset_commutative(s1_1, s1_2);
    vstd::seq_lib::lemma_multiset_commutative(s1_1, s1_3);

    vstd::seq_lib::lemma_multiset_commutative(s2_1 + s2_2, s2_3);
    vstd::seq_lib::lemma_multiset_commutative(s2_1, s2_2);
    vstd::seq_lib::lemma_multiset_commutative(s2_1, s2_3);

    // assert(s1.to_multiset() == s1_1.to_multiset().add(s1_2.to_multiset()).add(s1_3.to_multiset()));
    // assert(s1.to_multiset() == s1_1.to_multiset().add(s1_3.to_multiset()).add(s1_2.to_multiset()));

    // assert(s1.remove(i) =~= s1_1 + s1_3);
    assert(s1.remove(i).to_multiset() == s1_1.to_multiset().add(s1_3.to_multiset()));
    assert(s2.remove(j).to_multiset() == s2_1.to_multiset().add(s2_3.to_multiset()));

    // assert(s1_1.to_multiset().add(s1_3.to_multiset()).add(s1_2.to_multiset()) == (s1_1 + s1_3 + s1_2).to_multiset());
    // assert(s1_1 + s1_3 + s1_2 =~= (s1_1 + s1_3).push(s1[i]));

    (s1_1 + s1_3).to_multiset_ensures();
    (s2_1 + s2_3).to_multiset_ensures();

    assert(s1.remove(i).to_multiset().insert(s1[i]) == s1.to_multiset());
    assert(s2.remove(j).to_multiset().insert(s2[j]) == s2.to_multiset());

    assert(s1.remove(i).to_multiset() == s1.to_multiset().remove(s1[i]));
    assert(s2.remove(j).to_multiset() == s2.to_multiset().remove(s2[j]));
}

proof fn lemma_multiset_contains_transitive(s1: Seq<OperatorIndex>, s2: Seq<OperatorIndex>, s3: Seq<OperatorIndex>)
    requires
        multiset_contains(s1, s2),
        multiset_contains(s2, s3),
    
    ensures
        multiset_contains(s1, s3),
{
    reveal(multiset_contains);
}

proof fn lemma_multiset_contains_transitive_elem(s1: Seq<OperatorIndex>, s2: Seq<OperatorIndex>, op: OperatorIndex)
    requires
        multiset_contains(s1, s2),
        s2.contains(op),
    
    ensures
        s1.contains(op),
{
    reveal(multiset_contains);
    s1.to_multiset_ensures();
    s2.to_multiset_ensures();
    let _ = s1.to_multiset().count(op);
}

/**
 * Constructs the convergence trace in theorem_bounded_confluence
 */
#[verifier(opaque)]
spec fn bounded_confluence_trace(trace1: AugmentedTrace, trace2_configs: Seq<Configuration>, trace2_operators: Seq<OperatorIndex>) -> AugmentedTrace
    // recommends
    //     trace1.valid(),
    //     trace2_configs.len() == trace2_operators.len() + 1,
        
    //     (forall |i: int| 0 <= i < trace2_configs.len() ==> (#[trigger] trace2_configs[i].valid())) &&
    //     (forall |i: int| 0 <= i < trace2_operators.len() ==>
    //         (#[trigger] trace2_configs[i]).fireable(trace2_operators[i]) &&
    //         trace2_configs[i + 1] == trace2_configs[i].step(trace2_operators[i])),
            
    //     trace2_configs.first() == trace1.configs.first().config,

    //     multiset_contains(trace1.operators, trace2_operators),

    decreases trace2_operators.len()
{
    if trace2_operators.len() == 0 {
        trace1
    } else {
        let first_occurrence = trace1.operators.index_of_first(trace2_operators.first()).get_Some_0();

        let trace_prefix = trace1.take(first_occurrence + 1);
        let trace_suffix = trace1.skip(first_occurrence + 1);

        let commuted_trace_prefix = consistent_trace_commutes(trace_prefix);

        let convergent_trace_rest = bounded_confluence_trace(
            commuted_trace_prefix.drop_first().add(trace_suffix),
            trace2_configs.drop_first(), trace2_operators.drop_first(),
        );

        convergent_trace_rest.prepend(commuted_trace_prefix.operators.first(), commuted_trace_prefix.configs.first())
    }
}

/**
 * Theorem: If trace1 and trace2 have the same initial configuration,
 * and trace2.operators is contained in trace1.operators (counting multiplicity),
 * then trace2's final configuration converges to trace1 via a consistent trace.
 */
proof fn theorem_bounded_confluence(trace1: AugmentedTrace, trace2_configs: Seq<Configuration>, trace2_operators: Seq<OperatorIndex>)
    requires
        trace1.valid(),
        
        // (trace2_configs, trace2_operators) is a valid trace (without augmentation)
        trace2_configs.len() == trace2_operators.len() + 1,
        (forall |i: int| 0 <= i < trace2_configs.len() ==> (#[trigger] trace2_configs[i].valid())) &&
        (forall |i: int| 0 <= i < trace2_operators.len() ==>
            (#[trigger] trace2_configs[i]).fireable(trace2_operators[i]) &&
            trace2_configs[i + 1] == trace2_configs[i].step(trace2_operators[i])),
    
        trace2_configs.first() == trace1.configs.first().config,

        multiset_contains(trace1.operators, trace2_operators),

    ensures
        ({
            // Exists a valid augmented trace from trace2_config.last() to trace1.configs.last()
            let convergent_trace = bounded_confluence_trace(trace1, trace2_configs, trace2_operators);

            convergent_trace.valid() &&
            convergent_trace.len() == trace1.len() &&

            // trace2_configs is a prefix of convergent_trace
            (forall |i: int| 0 <= i < trace2_configs.len() ==>
                convergent_trace.configs[i].config == #[trigger] trace2_configs[i]) &&
            (forall |i: int| 0 <= i < trace2_operators.len() ==>
                convergent_trace.operators[i] == #[trigger] trace2_operators[i]) &&

            // convergent_trace starts and ends the same as trace1 (with the same augmentation)
            convergent_trace.configs.first() == trace1.configs.first() &&
            convergent_trace.configs.last() == trace1.configs.last()
        }),

    decreases trace2_operators.len()
{
    hide(Configuration::valid);
    hide(AugmentedConfiguration::valid);

    if trace2_operators.len() > 0 {
        let trace1_first_op = trace1.operators.first();
        let trace2_first_op = trace2_operators.first();

        assert(trace1.len() >= trace2_operators.len() > 0) by {
            lemma_multiset_contains_length(trace1.operators, trace2_operators);
        }

        let first_occurrence_opt = trace1.operators.index_of_first(trace2_first_op);
        let first_occurrence = first_occurrence_opt.get_Some_0();
        
        trace1.operators.index_of_first_ensures(trace2_first_op);
        assert(!first_occurrence_opt.is_None()) by {
            lemma_multiset_contains_transitive_elem(trace1.operators, trace2_operators, trace2_first_op);
        }

        let trace_prefix = trace1.take(first_occurrence + 1);
        let trace_suffix = trace1.skip(first_occurrence + 1);

        // assert(trace_prefix.operators.last() == trace2_first_op);

        let commuted_trace_prefix = consistent_trace_commutes(trace_prefix);
        lemma_consistent_trace_commutes(trace_prefix);

        let commuted_trace_rest = commuted_trace_prefix.drop_first().add(trace_suffix);

        let convergent_trace_rest = bounded_confluence_trace(commuted_trace_rest, trace2_configs.drop_first(), trace2_operators.drop_first());

        // commuted_trace_rest.operators = commuted_trace_prefix.operators[1:] + trace_suffix
        //                               = trace_prefix[:-1] + trace_suffix
        //                              >= trace2_operators[1:]
        assert(multiset_contains(commuted_trace_rest.operators, trace2_operators.drop_first())) by {
            assert(trace_prefix.operators.drop_last() + trace_suffix.operators =~= trace1.operators.remove(first_occurrence));
            lemma_multiset_contains_remove(trace1.operators, trace2_operators, first_occurrence, 0);
        }

        theorem_bounded_confluence(commuted_trace_rest, trace2_configs.drop_first(), trace2_operators.drop_first());

        assert(seq![trace2_configs[0]] + trace2_configs.drop_first() =~= trace2_configs);
        assert(seq![trace2_operators[0]] + trace2_operators.drop_first() =~= trace2_operators);
    }
    
    reveal(bounded_confluence_trace);
}

} // verus!
